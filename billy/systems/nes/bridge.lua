-- billy_bridge.lua
--
-- Lock-step file-IPC bridge between FCEUX and the Billy Mitchell (Python) brain.
--
-- Protocol (binary, little-endian):
--   state.bin  (Lua -> Python):  req_id(u32) frame(u32) done(u8) + 2048 bytes of work RAM
--   action.bin (Python -> Lua):  req_id(u32) command(u8) nsteps(u8)
--                                 then nsteps * [ dur(u16) buttonmask(u8) ]
--
-- The bridge is deliberately "dumb": it never decodes game meaning. It dumps all 2KB of
-- NES work RAM and lets Python do every bit of perception (testable, no Lua game logic).
--
-- Why it idles instead of truly pausing while Python thinks: FCEUX aborts a Lua script
-- that runs too long without yielding, so we cannot busy-spin for the seconds an LLM call
-- takes. Instead, while waiting for the next action we hold a NEUTRAL input and
-- frame-advance, so Mario stands still in a safe spot. Reflex exchanges are sub-millisecond
-- (~0 idle frames); only the occasional LLM decision idles for a moment.

local RUNTIME = os.getenv("BILLY_RUNTIME") or "/tmp/billy"
local STATE     = RUNTIME .. "/state.bin"
local STATE_TMP = RUNTIME .. "/state.bin.tmp"
local ACTION    = RUNTIME .. "/action.bin"
local RAM_SIZE  = 0x800

-- bit0..bit7 -> the key names FCEUX's joypad.set expects.
local BUTTON_ORDER = { "A", "B", "select", "start", "up", "down", "left", "right" }

-- Commands (must match billy/config.py).
local CMD_RUN_PLAN, CMD_SAVESTATE, CMD_LOADSTATE, CMD_SOFT_RESET = 0, 1, 2, 3

local req_id = 0
-- Independent savestate slots: slot 0 = level-start (attempt reset); slot 1+ = micro-search
-- checkpoints (try a jump, rewind, try another). Created lazily.
local snaps = {}
local function get_snap(slot)
  if not snaps[slot] then snaps[slot] = savestate.create() end
  return snaps[slot]
end

-- Optional speed: BILLY_SPEED=normal|turbo|maximum (default normal so a human can watch).
local speed = os.getenv("BILLY_SPEED")
if speed and emu.speedmode then emu.speedmode(speed) end

-- DISABLE REWIND: Billy should not cheat by rewinding time during play.
-- This ensures fair gameplay - no second chances via time manipulation.
if emu.setrewind then
  emu.setrewind(false)  -- Disable FCEUX rewind feature
end

local function u32(n)
  return string.char(n % 256, math.floor(n / 256) % 256,
                     math.floor(n / 65536) % 256, math.floor(n / 16777216) % 256)
end

local function read_ram()
  local ok, s = pcall(memory.readbyterange, 0x0000, RAM_SIZE)
  if ok and type(s) == "string" and #s == RAM_SIZE then return s end
  local t = {}
  for a = 0, RAM_SIZE - 1 do t[a + 1] = string.char(memory.readbyte(a)) end
  return table.concat(t)
end

local function write_state(done)
  local body = u32(req_id) .. u32(emu.framecount()) .. string.char(done and 1 or 0) .. read_ram()
  local f = assert(io.open(STATE_TMP, "wb"))
  f:write(body); f:close()
  os.rename(STATE_TMP, STATE)  -- publish atomically
end

local NEUTRAL = {}  -- no buttons held

-- Block until action.bin echoes our current req_id. While blocked, hold neutral and
-- frame-advance so FCEUX stays alive and Mario idles in place. We RE-PUBLISH the state each
-- frame so it reappears if the Python brain starts late or clears the files (avoids a
-- startup deadlock); re-publishing the same req_id is idempotent on the Python side.
local function wait_for_action()
  while true do
    local f = io.open(ACTION, "rb")
    if f then
      local data = f:read("*a"); f:close()
      if data and #data >= 5 then  -- save/load/reset actions are 5 bytes (req+cmd, no plan)
        local rid = data:byte(1) + data:byte(2) * 256 + data:byte(3) * 65536 + data:byte(4) * 16777216
        if rid == req_id then return data end
      end
    end
    write_state(false)
    joypad.set(1, NEUTRAL)
    emu.frameadvance()
  end
end

local function mask_to_buttons(mask)
  local t = {}
  for bit = 0, 7 do
    if math.floor(mask / (2 ^ bit)) % 2 >= 1 then t[BUTTON_ORDER[bit + 1]] = true end
  end
  return t
end

-- Execute a button plan: nsteps groups of (duration_frames, button bitmask).
local function run_plan(data)
  local nsteps = data:byte(6)
  local pos = 7
  for _ = 1, nsteps do
    local dur = data:byte(pos) + data:byte(pos + 1) * 256
    local mask = data:byte(pos + 2)
    pos = pos + 3
    local buttons = mask_to_buttons(mask)
    for _ = 1, dur do
      joypad.set(1, buttons)
      emu.frameadvance()
    end
  end
end

emu.print("[billy_bridge] online. runtime=" .. RUNTIME)

while true do
  req_id = req_id + 1
  write_state(false)

  -- HUD for the spike: show what the bridge sees (Mario world-X straight from RAM).
  local mario_x = memory.readbyte(0x6D) * 256 + memory.readbyte(0x86)
  gui.text(8, 8, string.format("BILLY req=%d x=%d", req_id, mario_x))

  local data = wait_for_action()
  local cmd = data:byte(5)
  if cmd == CMD_SAVESTATE then
    savestate.save(get_snap(data:byte(6) or 0))
  elseif cmd == CMD_LOADSTATE then
    savestate.load(get_snap(data:byte(6) or 0))
  elseif cmd == CMD_SOFT_RESET then
    emu.softreset()
  else
    run_plan(data)
  end
end
