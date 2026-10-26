-- settings for wulkanium: dlss mode and screen effects. the renderer lists its effects in
-- data/wulkanium_effects.json; this creates their convars (wk_fx_<effect>, wk_fx_<effect>_<param>)
-- and a page under Options > wulkanium. wk_fx_reload re-reads the list after adding effects.

local effects = {}

local function load()
	effects = util.JSONToTable(file.Read("wulkanium_effects.json", "DATA") or "") or {}
	for _, e in ipairs(effects) do
		local base = "wk_fx_" .. e.key
		CreateClientConVar(base, e.enabled and "1" or "0", true, false, "enable " .. e.name, 0, 1)
		for _, p in ipairs(e.params or {}) do
			CreateClientConVar(base .. "_" .. p.name, tostring(p.default), true, false, e.name .. ": " .. p.name, p.min, p.max)
		end
	end
	return #effects
end

load()

concommand.Add("wk_fx_reload", function()
	print("wulkanium: " .. load() .. " effects")
end)

concommand.Add("wk_fx_list", function()
	for _, e in ipairs(effects) do
		local base = "wk_fx_" .. e.key
		print(string.format("%-28s %s", base, GetConVar(base):GetBool() and "on" or "off"))
		for _, p in ipairs(e.params or {}) do
			print(string.format("  %-26s %s", base .. "_" .. p.name, GetConVar(base .. "_" .. p.name):GetString()))
		end
	end
end)

-- dlss quality (dlss builds; End toggles dlss itself)
CreateClientConVar("wk_dlss_mode", "0", true, false, "DLSS mode: 0 DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra Performance", 0, 4)
CreateClientConVar("wk_dlss_mip_bias", "0", true, false, "added to DLSS's texture mip bias; positive = less sharp, less shimmer", -1, 1)
CreateClientConVar("wk_ggx_debug", "0", false, false, "SurfaceGGX debug view: 1 albedo, 2 normal, 3 roughness, 4 metal, 5 diffuse light, 6 direct specular, 7 reflection, 8 ambient cube, 9 raw mask, 10 raw bump, 11 uv", 0, 11)

hook.Add("PopulateToolMenu", "wulkanium_dlss", function()
	spawnmenu.AddToolMenuOption("Options", "wulkanium", "wulkanium_dlss", "DLSS", "", "", function(panel)
		panel:ClearControls()
		local box = panel:ComboBox("Mode", "wk_dlss_mode")
		for i, name in ipairs({ "DLAA (native)", "Quality", "Balanced", "Performance", "Ultra Performance" }) do
			box:AddChoice(name, i - 1)
		end
	end)
end)

hook.Add("PopulateToolMenu", "wulkanium_effects", function()
	spawnmenu.AddToolMenuOption("Options", "wulkanium", "wulkanium_effects", "Screen effects", "", "", function(panel)
		panel:ClearControls()
		if #effects == 0 then
			panel:Help("No effects found. Put .wgsl files in garrysmod/shaders/wulkanium and run wk_fx_reload.")
			return
		end
		for _, e in ipairs(effects) do
			local base = "wk_fx_" .. e.key
			panel:CheckBox(e.name, base)
			for _, p in ipairs(e.params or {}) do
				panel:NumSlider("    " .. p.name, base .. "_" .. p.name, p.min, p.max, p.int and 0 or 2)
			end
		end
	end)
end)
