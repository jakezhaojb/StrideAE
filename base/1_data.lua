print '==> loading dataset'

if config.dataset == 'cifar' then
   print "CIFAR loading."
   local loaded = torch.load(path_to_training)
   images = loaded.data:reshape(50000,3,32,32)
   images = images:type('torch.FloatTensor')
else
   error('Not this dataset')
end

collectgarbage()
