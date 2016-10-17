-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
local path = require 'paths'

local cmd = torch.CmdLine()
cmd:option('-filePath', 'MINC_paths.txt')
local opt = cmd:parse(arg)

-- Load FMD dataset
local loader = require "loader"
local XTrain = loader:load_fmd(opt.filePath):float()
local N = XTrain:size(1)
torch.save('gray_scale.dat', Xtrain)
