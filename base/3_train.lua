print '==> defining training procedure'

parameters, gradParameters = model:getParameters()

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- training numbers
   trSize = images:size(1)

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trSize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [config.batchSize = ' .. config.batchSize .. ']')
   for t = 1,trSize,config.batchSize do
      -- disp progress
      xlua.progress(t, trSize)

      -- create mini batch
      local inputs = torch.Tensor(config.batchSize, images:size(2), images:size(3), images:size(4)):cuda()
      local targets =  torch.Tensor(config.batchSize, images:size(2), images:size(3), images:size(4)):cuda()
      for i = t,math.min(t+config.batchSize-1,trSize) do
         -- load new sample
         local input = images[shuffle[i]]:cuda()
         inputs[i-t+1]:copy(input)
         targets[i-t+1]:copy(input)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                       gradParameters:zero()
                       local output = model:forward(inputs)
                       local err = criterion:forward(output, targets)
                       --print(output:max())  -- Good point for DEBUGING.
                       f = err
                       -- estimate df/dW
                       local df_do = criterion:backward(output, targets)
                       model:backward(inputs, df_do)
                       -- normalize gradients and f(X)
                       -- return f and df/dX
                       return f,gradParameters
                    end

      optim.sgd(feval, parameters, config.optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trSize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- model saving
   if epoch % 20 == 0 then
      os.execute('mkdir -p ' .. config.save_path)
      local filename = paths.concat(config.save_path, 'model_net')
      print('==> saving model to '..filename)
      torch.save(filename, model)
   end
   print(string.format("Err: %.2f", f))
   windows = windows or {}
   windows.enc = gfx.image(weights.enc, {legend='encoder', zoom=4, win=windows.enc})
   windows.dec = gfx.image(weights.dec:transpose(1,2), {legend='decoder', zoom=4, win=windows.dec})
   epoch = epoch + 1
end
