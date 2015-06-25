print '==> defining training procedure'

parameters, gradParameters = model:getParameters()

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trSize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trSize,batchSize do
      -- disp progress
      xlua.progress(t, trSize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trSize) do
         -- load new sample
         local input = images[shuffle[i]]:cuda()
         input = input:type('torch.FloatTensor'):reshape(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
         local target = input:clone()
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                       gradParameters:zero()
                       local f = 0
                       local Nf = 0  -- normalized error
                       for i = 1,#inputs do
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          Nf = Nf + math.sqrt(err)/targets[i]:norm()
                          --print(output:max())  -- Good point for DEBUGING.
                          f = f + err
                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)
                       end
                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs
                       Nf = Nf/#inputs
                       print(string.format("Nerr: %.2f, ", Nf) .. string.format("Err: %.2f", f))

                       -- return f and df/dX
                       return f,gradParameters
                    end

      optim.sgd(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trSize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   -- model saving
   if epoch % 20 == 0 then
      os.execute('mkdir -p ' .. save_path)
      local filename = paths.concat(save_path, 'model_net')
      print('==> saving model to '..filename)
      torch.save(filename, model)
   end
   gfx.image(weights.enc, {legend='', zoom=4, win=enc})
   gfx.image(weights.dec:transpose(1,2), {legend='', zoom=4, win=dec})
   epoch = epoch + 1
end