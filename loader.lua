require 'torch'
require 'image'

local pwd = '/Users/art/datasets/materials_textures/materials_textures/fmd/images_cropped_256/'

local classes = {'fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood'}
local type = {'moderate', 'object'}

-- convert rgb to grayscale by averaging channel intensities
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

-- str = string.format("%03d", 49)
-- print(str)

local loader = {}

function loader:load_fmd()
  all_images = torch.Tensor(1000,3,256,256)
  grayscale_images = torch.Tensor(1000,256,256)

  print (sys.COLORS.red .. '==> loading images')

  iter = 1
  for i, name_class in ipairs(classes) do
    if name_class == 'foliage' then
      for k = 1,100 do
        all_images[iter] = image.load(pwd .. name_class .. '/' .. name_class .. '_final_' .. string.format("%03d", k) .. '_new.jpg')
        iter = iter + 1
        -- print(iter)
      end
    else
      for j, name_type in ipairs(type) do
        for k = 1,50 do
          -- path = pwd .. name_class .. '/' .. name_class .. '_' .. name_type .. '_' .. string.format("%03d", k) .. '_new.jpg'
          -- print(path)
          -- print(image.load(path):size())
          all_images[iter] = image.load(pwd .. name_class .. '/' .. name_class .. '_' .. name_type .. '_' .. string.format("%03d", k) .. '_new.jpg')
          -- print(iter)
          iter = iter + 1
        end
      end
    end
  end

  -- convert rgb images to grayscale
  for i = 1,1000 do
    grayscale_images[i] = rgb2gray(all_images[i])
  end

  -- itorch.image(grayscale_images[100])
  -- itorch.image(grayscale_images[300])

  print('done!')
  return grayscale_images
end

return loader
