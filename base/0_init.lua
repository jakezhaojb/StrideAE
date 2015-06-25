--require 'cutorch'
--require 'cunn'
require 'cudnn'
require 'nnx'
require 'xlua'
require 'optim'
gfx = require 'gfx.js'

-- This file involves all kinds of configurations!!!
--
cutorch.setDevice(1)
debugFlag = false
torch.setdefaulttensortype('torch.FloatTensor')

dataset = 'cifar'
trSize = 60000 -- for MNIST
if debugFlag then
   trSize = 1000
end

filterSize = 9
nOutplane = 32
poolSize = 4
l1weight = 0 -- To be tunned
init_scale_down = 1

optimState = {
   learningRate = 0.0005,
   weightDecay = 0.00001,
   momentum = 0.9,
   learningRateDecay = 5e-4
}
batchSize = 256

dofile("./Modules/init.lua")
maxPoolFlag = true
paraTied = true

if dataset == 'cifar' then
   nInplane = 3
   path_to_training = '../Data/cifar/CIFAR_CN_train.t7'
   path_to_testing = '../Data/cifar/CIFAR_CN_test.t7'
else
   error("No dataset is found.")
end

poolBeta = 100

nEpoches = 20

local filename = path.basename(paths.dirname(paths.thisfile()))
save_path = paths.concat('../Results', filename)
