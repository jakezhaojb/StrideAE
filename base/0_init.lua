--require 'cutorch'
--require 'cunn'
require 'cudnn'
require 'cunn'
require 'nnx'
require 'xlua'
require 'optim'
require 'myaenn'
dofile("./Modules/init.lua")
gfx = require 'gfx.js'

-- General configurations
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- Data
config.dataset = 'cifar'
if config.dataset == 'cifar' then
   nInplane = 3
   path_to_training = '../Data/cifar/CIFAR_CN_train.t7'
   path_to_testing = '../Data/cifar/CIFAR_CN_test.t7'
else
   error("No dataset is found.")
end

-- Model
config.filterSize = 5
config.nOutplane = 32
config.poolSize = 4
config.poolBeta = 100
config.l1weight = 10 -- To be tunned
config.init_scale_down = 0.1

-- Train
config.optimState = {
   learningRate = 0.1,
   weightDecay = 5e-5,
   momentum = 0.9,
   learningRateDecay = 5e-4
}
config.batchSize = 256
config.nEpoches = 5
local filename = path.basename(paths.dirname(paths.thisfile()))
config.save_path = paths.concat('../Results', filename)
