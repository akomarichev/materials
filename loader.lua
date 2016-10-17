require 'torch'
require 'image'

-- convert rgb to grayscale by averaging channel intensities
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

-- str = string.format("%03d", 49)
-- print(str)

local loader = {}

function loader:load_fmd(filename)
  local path = "/opt/home/datasets/materials_textures/" .. filename

  local nOfLines = 0
  for _ in io.lines(path) do
    nOfLines = nOfLines + 1
  end

  local file = io.open(path)

  print("#ofLines: " .. nOfLines)

  grayscale_images = torch.Tensor(nOfLines,256,256)

  print (sys.COLORS.red .. '==> loading images')

  iter = 1
  for line in file:lines() do
    grayscale_images[iter] = rgb2gray(image.load(line))
    iter = iter + 1
  end

  print('Max grayscale: ' .. torch.max(grayscale_images))
  print('Min grayscale: ' .. torch.min(grayscale_images))

  print(sys.COLORS.red .. 'Loaded!')
  return grayscale_images
end

return loader
