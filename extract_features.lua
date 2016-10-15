local csv = require 'csvigo'
local path = require 'paths'
require 'nn'
require 'torch'
require 'image'

tempfilename = "ae_features.csv"

-- saving model
modelName = '/AE_model.net'
filename = path.cwd() .. modelName

if path.filep(filename) then
  print("Model exists!")
  model = torch.load(filename)
  autoencoder = model:get(1)
  print(autoencoder)
end

function rgb2gray(im)
	-- Image.rgb2y uses a different weight mixture

	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	if dim ~= 3 then
		 print('<error> expected 3 channels')
		 return im
	end

	-- a cool application of tensor:select
	local r = im:select(1, 1):csub(124)
	local g = im:select(1, 2):csub(117)
	local b = im:select(1, 3):csub(104)

	local z = torch.Tensor(w, h):zero()

	-- z = z + 0.21r
	z = z:add(0.21, r)
	z = z:add(0.72, g)
	z = z:add(0.07, b)
	return z
end

local img
local temp
local output

-- local loader = require "loader"
-- local XTrain = loader:load_fmd('/Users/art/datasets/materials_textures/materials_textures/fmd/images_cropped_256/'):float():div(255)
-- autoencoder:forward(XTrain)

csvf = csv.File(tempfilename, "w")
local file = io.open("/opt/home/datasets/materials_textures/fmd/FMD_paths.txt")
if file then
    for line in file:lines() do
      --line = '/Users/art/datasets/materials_textures/materials_textures/fmd/images_cropped_256/fabric/fabric_moderate_001_new.jpg'
      --print(line)
      temp = torch.Tensor(1,256,256)
      img = image.load(line)
      temp[1] = rgb2gray(img)
      temp = temp:float():div(255)
      --print(#temp)
      output = autoencoder:forward(temp)
      --print(output:size())
      output = output:view(-1)
      --print(output:size())
      list = output:totable()
      --print(#list)
      table.insert(list, 1, line)
      csvf:write(list)
    end
else
end
csvf:close()

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
