-- benchmark helper: save a spot per map, lock the player there, time frames.
-- commands: bench_save, bench_lock, bench_unlock, bench_run [seconds] [warmup]
AddCSLuaFile()

if SERVER then
	local function allowed(ply)
		return IsValid(ply) and (game.SinglePlayer() or ply:IsAdmin())
	end

	concommand.Add("wulk_bench_place", function(ply, _, args)
		if not allowed(ply) then return end
		local n = function(i) return tonumber(args[i]) or 0 end
		ply:SetMoveType(MOVETYPE_NONE)
		ply:SetPos(Vector(n(1), n(2), n(3)))
		ply:SetEyeAngles(Angle(n(4), n(5), 0))
		ply:SetLocalVelocity(vector_origin)
	end)

	concommand.Add("wulk_bench_release", function(ply)
		if allowed(ply) then ply:SetMoveType(MOVETYPE_WALK) end
	end)
	return
end

local FILE = "wulkanium_bench.json"
local spots = util.JSONToTable(file.Read(FILE, "DATA") or "") or {}
local locked -- view angles while locked

local function current()
	return spots[game.GetMap()]
end

concommand.Add("bench_save", function()
	local ply = LocalPlayer()
	local p, a, e = ply:GetPos(), ply:EyeAngles(), ply:EyePos()
	spots[game.GetMap()] = { pos = { p.x, p.y, p.z }, ang = { a.p, a.y } }
	file.Write(FILE, util.TableToJSON(spots, true))
	print(string.format("bench: saved %s  pos %.1f %.1f %.1f  eye %.1f %.1f %.1f  ang %.2f %.2f",
		game.GetMap(), p.x, p.y, p.z, e.x, e.y, e.z, a.p, a.y))
end)

concommand.Add("bench_lock", function()
	local s = current()
	if not s then
		print("bench: no spot saved for " .. game.GetMap() .. ", use bench_save first")
		return
	end
	locked = Angle(s.ang[1], s.ang[2], 0)
	RunConsoleCommand("wulk_bench_place", s.pos[1], s.pos[2], s.pos[3], s.ang[1], s.ang[2])
	print("bench: locked (bench_unlock to release)")
end)

concommand.Add("bench_unlock", function()
	locked = nil
	RunConsoleCommand("wulk_bench_release")
	print("bench: unlocked")
end)

-- hold the view and drop movement while locked
hook.Add("CreateMove", "wulkanium_bench", function(cmd)
	if not locked then return end
	cmd:ClearMovement()
	cmd:ClearButtons()
	cmd:SetViewAngles(locked)
end)

concommand.Add("bench_run", function(_, _, args)
	local secs = tonumber(args[1]) or 10
	local warmup = tonumber(args[2]) or 2
	local times = {}
	local last = SysTime()
	local begin = last + warmup
	local stop = begin + secs
	print(string.format("bench: %ds run after %ds warmup...", secs, warmup))

	hook.Add("PostRender", "wulkanium_bench_run", function()
		local now = SysTime()
		if now >= begin then times[#times + 1] = now - last end
		last = now
		if now < stop then return end
		hook.Remove("PostRender", "wulkanium_bench_run")

		local n, total = #times, 0
		for _, t in ipairs(times) do total = total + t end
		table.sort(times, function(a, b) return a > b end)
		local worst = math.max(1, math.floor(n / 100))
		local low = 0
		for i = 1, worst do low = low + times[i] end
		low = low / worst

		local msg = string.format("bench: %s  %d frames  avg %.1f fps (%.2f ms)  1%% low %.1f fps  worst %.2f ms",
			game.GetMap(), n, n / total, total / n * 1000, 1 / low, times[1] * 1000)
		print(msg)
		chat.AddText(Color(120, 200, 255), msg)
		file.Append("wulkanium_bench_results.txt", os.date("%Y-%m-%d %H:%M:%S  ") .. msg .. "\n")
	end)
end)
