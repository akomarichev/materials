require 'torch'
require 'image'
local path = require 'paths'

local path = "/opt/home/datasets/materials_textures/FMD_paths.txt"
local file = io.open(path)

function rgb2gray(im)
	-- Image.rgb2y uses a different weight mixture

	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	if dim ~= 3 then
		 print('<error> expected 3 channels')
		 return im
	end

	-- a cool application of tensor:select
	local r = im:select(1, 1) --:csub(124/256)
	local g = im:select(1, 2) --:csub(117/256)
	local b = im:select(1, 3) --:csub(104/256)

	local z = torch.Tensor(w, h):zero()

	-- z = z + 0.21r
	z = z:add(0.21, r)
	z = z:add(0.72, g)
	z = z:add(0.07, b)
	return z
end

iter = 1
for line in file:lines() do
  local im = rgb2gray(image.load(line))
  iter = iter + 1
  image.save('filters_enc_l1_' .. iter .. '.jpg', im)
  if iter == 3 then break end
end
