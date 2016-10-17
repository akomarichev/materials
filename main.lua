-- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
local path = require 'paths'

print(cuda)

-- if cuda then
--   print('Number of GPUs: ' .. cutorch.getDeviceCount())
--   gpus = torch.range(1, cutorch.getDeviceCount()):totable()
-- end

-- Set up Torch
print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end

-- Load MNIST data
-- local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
-- local XTest = mnist.testdataset().data:float():div(255)
-- local N = XTrain:size(1)
-- if cuda then
--   XTrain = XTrain:cuda()
--   XTest = XTest:cuda()
-- end

-- itorch.image(Xtrain[100])
-- itorch.image(Xtrain[300])

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'AE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AdvAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-filePath', 'MINC_paths.txt')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-nOfHiddenUnits', 100)
local opt = cmd:parse(arg)
opt.batchSize = 50 -- Currently only set up for divisors of N

-- Load FMD dataset
local loader = require "loader"
local XTrain = loader:load_fmd(opt.filePath):float()
local N = XTrain:size(1)
torch.save('gray_scale.dat', Xtrain)
print(XTrain:size())
if cuda then
  XTrain = XTrain:cuda()
  -- XTrain = nn.DataParallelTable(1):add(XTrain, gpus):cuda()
end

-- Create model
local Model = require ('models/' .. opt.model)
Model:createAutoencoder(XTrain, opt.nOfHiddenUnits)
local autoencoder = Model.autoencoder
if cuda then
  autoencoder:cuda()
  --autoencoder = nn.DataParallelTable(1):add(autoencoder, gpus):cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(autoencoder, cudnn)
  end
end

print("done 2!")
print(autoencoder)

-- Create adversary (if needed)
-- local adversary
-- if opt.model == 'AdvAE' then
--   Model:createAdversary()
--   adversary = Model.adversary
--   if cuda then
--     adversary:cuda()
--     -- Use cuDNN if available
--     if hasCudnn then
--       cudnn.convert(adversary, cudnn)
--     end
--   end
-- end

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
-- local thetaAdv, gradThetaAdv
-- if opt.model == 'AdvAE' then
--   thetaAdv, gradThetaAdv = adversary:getParameters()
-- end

print("Number of parameters: " .. theta:size(1))

-- Create loss
local criterion = nn.BCECriterion()
if cuda then
  criterion:cuda()
  --criterion = nn.DataParallelTable(1):add(simple_criterion, gpus):cuda()
end

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()
  if opt.model == 'AdvAE' then
    gradThetaAdv:zero()
  end

  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  -- Regularization phase
  if opt.model == 'Seq2SeqAE' then
    -- Clamp RNN gradients to prevent exploding gradients
    gradTheta:clamp(-10, 10)
  elseif opt.model == 'VAE' then
    local encoder = Model.encoder

    -- Optimize Gaussian KL-Divergence between inference model and prior: DKL(q(z)||N(0, I)) = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
    local nElements = xHat:nElement()
    local q = encoder.output
    local mean, logVar = table.unpack(encoder.output)
    local var = torch.exp(logVar)
    local KLLoss = -0.5 * torch.sum(1 + logVar - torch.pow(mean, 2) - var)
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}  -- Normalise gradient of loss (same normalisation as BCECriterion)
    encoder:backward(x, gradKLLoss)
  elseif opt.model == 'AdvAE' then
    local encoder = Model.encoder
    local real = torch.Tensor(opt.batchSize, Model.zSize):normal(0, 1):typeAs(XTrain) -- Real samples ~ N(0, 1)
    local YReal = torch.ones(opt.batchSize):typeAs(XTrain) -- Labels for real samples
    local YFake = torch.zeros(opt.batchSize):typeAs(XTrain) -- Labels for generated samples

    -- Train adversary to maximise log probability of real samples: max_D log(D(x))
    local pred = adversary:forward(real)
    local realLoss = criterion:forward(pred, YReal)
    local gradRealLoss = criterion:backward(pred, YReal)
    adversary:backward(real, gradRealLoss)

    -- Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
    pred = adversary:forward(encoder.output)
    local fakeLoss = criterion:forward(pred, YFake)
    advLoss = realLoss + fakeLoss
    local gradFakeLoss = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(encoder.output, gradFakeLoss)

    -- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
    local minimaxLoss = criterion:forward(pred, YReal)
    loss = loss + minimaxLoss
    local gradMinimaxLoss = criterion:backward(pred, YReal)
    local gradMinimax = adversary:updateGradInput(encoder.output, gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
    encoder:backward(x, gradMinimax)
  end

  return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end

  return advLoss, gradThetaAdv
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = opt.learningRate}
local advOptimParams = {learningRate = opt.learningRate}
local __, loss
local losses, advLosses = {}, {}

for epoch = 1, opt.epochs do
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, N, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    __, loss = optim[opt.optimiser](feval, theta, optimParams)
    losses[#losses + 1] = loss[1]

    -- Train adversary
    if opt.model == 'AdvAE' then
      __, loss = optim[opt.optimiser](advFeval, thetaAdv, advOptimParams)
      advLosses[#advLosses + 1] = loss[1]
    end
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  if opt.model == 'AdvAE' then
    plots[#plots + 1] = {'Adversary', torch.linspace(1, #advLosses, #advLosses), torch.Tensor(advLosses), '-'}
  end
  gnuplot.pngfigure('Training.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()
end


-- saving model
modelName = '/AE_model.net'
filename = path.cwd() .. modelName

if path.filep(filename) then
  print("Model exists!")
  model = torch.load(filename)
  print(model)
else
  print("Model does not exist! Needs to be trained first.")
  -- train()
  torch.save(filename, autoencoder)
end

-- Test
-- print('Testing')
-- x = XTest:narrow(1, 1, 10)
-- local xHat
-- if opt.model == 'DenoisingAE' then
--   -- Normally this should be switched to evaluation mode, but this lets us extract the noised version
--   xHat = autoencoder:forward(x)
--
--   -- Extract noised version from denoising AE
--   x = Model.noiser.output
-- else
--   autoencoder:evaluate()
--   xHat = autoencoder:forward(x)
-- end

-- Plot reconstructions
-- image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))

-- Plot samples
-- if opt.model == 'VAE' or opt.model == 'AdvAE' then
--   local decoder = Model.decoder
--   local height, width = XTest:size(2), XTest:size(3)
--   local samples = torch.Tensor(15 * height, 15 * width):typeAs(XTest)
--   local std = 0.05
--
--   -- Sample 15 x 15 points
--   for i = 1, 15  do
--     for j = 1, 15 do
--       local sample = torch.Tensor({2 * i * std - 16 * std, 2 * j * std - 16 * std}):typeAs(XTest):view(1, 2) -- Minibatch of 1 for batch normalisation
--       samples[{{(i-1) * height + 1, i * height}, {(j-1) * width + 1, j * width}}] = decoder:forward(sample)
--     end
--   end
--   image.save('Samples.png', samples)
-- end
