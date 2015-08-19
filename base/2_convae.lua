-- Convolutional Autoencoder
-- Framework1: valid + pad1 + valid or valid + pad2 + valid -> both are only capable to reconstruct the central part of input
-- Framework2: same + same -> capable to reconstruct the whole image; same = pad1 + valid


conv_autoencoder = function()

   local pad1 = (config.filterSize - 1) / 2

   local encoder = nn.Sequential()
   encoder:add(cudnn.SpatialConvolution(nInplane, config.nOutplane, config.filterSize, config.filterSize, 1, 1, pad1, pad1))
   encoder:add(cudnn.ReLU())

   local decoder = nn.Sequential()
   decoder:add(cudnn.NormSpatialConvolution(config.nOutplane, nInplane, config.filterSize, config.filterSize, 1, 1, pad1, pad1))

   if config.init_scale_down then
      print "==> scaling down the initialized weights."
      decoder:get(1).weight:mul(config.init_scale_down)
   end

   local conv_ae = nn.Sequential()
   conv_ae:add(encoder)

   -- Using Upsampling MaxPooling and Unpooling
   conv_ae:add(nn.MaxPoolUnpool(poolSize, poolSize))

   conv_ae:add(nn.L1Penalty(l1weight, true))
   conv_ae:add(decoder)

   local criterion = nn.MSECriterion()
   criterion.sizeAverage = true

   conv_ae:cuda()
   criterion:cuda()

   local weights = {}
   weights = {enc = encoder:get(1).weight,
              dec = decoder:get(1).weight}

   return conv_ae, criterion, weights
end

model, criterion, weights = conv_autoencoder()
