local csv = require 'csvigo'
local path = require 'paths'
require 'torch'

tempfilename = "ae_features.csv"

temp = torch.Tensor(5)

print(temp:totable())

csvf = csv.File(tempfilename, "w")
local file = io.open("/opt/home/datasets/materials_textures/fmd/FMD_paths.txt")
if file then
    for line in file:lines() do
      list = temp:totable()
      table.insert(list, 1, line)
      csvf:write(list)
    end
else
end
csvf:close()

-- saving model
modelName = '/AE_model.net'
filename = path.cwd() .. modelName

print("Model exists!")
model = torch.load(filename)
print(filename)
print(model)

-- local combine = function(list, features)
--   for i,val in ipairs(features) do
--
--   end
--
--   return list
-- end

-- -- test writing file
-- function writeRecs(csvf)
--    csvf:write({"a","b","c"})
--    csvf:write({01, 02, 03})
--    csvf:write({11, 12, 13})
-- end
--
--
-- csvf = csv.File(tempfilename, "w")
-- writeRecs(csvf)
-- csvf:close()
