print '==> loading dataset'

if dataset == 'cifar' then
   print "CIFAR loading."
   local loaded = torch.load(path_to_training)
   images = loaded.datacn:reshape(50000,3,32,32)
   images = images:type('torch.FloatTensor')
   if trSize > images:size(1) then
      trSize = images:size(1)
   else
      images = images[{ {1, trSize}, {}, {}, {}  }]
   end
else
   error('Not this dataset')
end

collectgarbage()
