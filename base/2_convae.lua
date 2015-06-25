-- Convolutional Autoencoder
-- Framework1: valid + pad1 + valid or valid + pad2 + valid -> both are only capable to reconstruct the central part of input
-- Framework2: same + same -> capable to reconstruct the whole image; same = pad1 + valid


conv_autoencoder = function()

   local pad1 = (filterSize - 1) / 2

   local encoder = nn.Sequential()
   encoder:add(nn.SpatialPadding(pad1, pad1, pad1, pad1, 3, 4))
   encoder:add(cudnn.SpatialConvolution(nInplane, nOutplane, filterSize, filterSize))
   encoder:add(cudnn.ReLU())

   local decoder = nn.Sequential()
   decoder:add(nn.SpatialPadding(pad1, pad1, pad1, pad1, 3, 4))
   decoder:add(cudnn.NormSpatialConvolution(nOutplane, nInplane, filterSize, filterSize))

   if paraTied then
      decoder:get(2).weight = encoder:get(2).weight:transpose(1,2)
      decoder:get(2).gradWeight = encoder:get(2).gradWeight:transpose(1,2)
   end

   if init_scale_down ~= nil then
      print "==> scaling down the initialized weights."
      decoder:get(2).weight:mul(init_scale_down)
   end

   -- Remark: no need to flip the weights
   local conv_ae = nn.Sequential()
   conv_ae:add(encoder)

   conv_ae:add(nn.MaxPoolUnpool(poolSize, poolSize))

   conv_ae:add(nn.L1Penalty(l1weight))
   conv_ae:add(decoder)

   local criterion = nn.MSECriterion()
   criterion.sizeAverage = false

   conv_ae:cuda()
   criterion:cuda()

   local weights = {}
   weights = {enc = encoder:get(2).weight,
              dec = decoder:get(2).weight}

   return conv_ae, criterion, weights
end

model, criterion, weights = conv_autoencoder()
