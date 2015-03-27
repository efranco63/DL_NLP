-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Your submission should run on Mercer and contain: 
-- a completed TEAMNAME_A3_skeleton.lua,
--
-- a script TEAMNAME_A3_baseline.lua that is just the provided A3_baseline.lua modified
-- to use your TemporalLogExpPooling module instead of nn.TemporalMaxPooling,
--
-- a saved trained model from TEAMNAME_A3_baseline.lua for which you have done some basic
-- hyperparameter tuning on the training data,
-- 
-- and a script TEAMNAME_A3_gradientcheck.lua that takes as input from stdin:
-- a float epsilon, an integer N, N strings, and N labels (integers 1-5)
-- and prints to stdout the ratios |(FD_epsilon_ijk - exact_ijk) / exact_ijk|
-- where exact_ijk is the backpropagated gradient for weight ijk for the given input
-- and FD_epsilon_ijk is the second-order finite difference of order epsilon
-- of weight ijk for the given input.
------------------------------------------------------------------------

local TemporalLogExPooling, parent = torch.class('nn.TemporalLogExPooling', 'nn.Module')

function TemporalLogExPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExPooling:updateOutput(input) -- MODIFY THE NAME BACK TO TemporalLogExpPooling
   -----------------------------------------------
   
   -- if the input tensor is 2D (nInputFrame x inputFrameSize)
   if input:dim() ==  2 then

      -- nOutputFrame
      nOutputFrame = (input:size(1) - self.kW)/self.dW + 1
      -- Output tensor
      output = torch.Tensor(nOutputFrame, input:size(2)):fill(0)
      -- perform log exponential pooling
      iter = 1 --to keep track of what frame we are updating in output
      for i=1,input:size(1),self.dW do
         -- will store the summation of the exponents
         s = torch.Tensor(1,input:size(2)):fill(0)
         -- calculate the summation of the exponents and store in s
         if (i+self.kW-1) <= input:size(1) then --if what the kernel envelopes is not outside the limit
            for j=1,i+self.kW-1 do
               -- create a copy of the input so we won't modify the input values
               copyt = torch.Tensor(input[{ {j},{} }]:size()):copy(input[{ {j},{} }])
               s:add(torch.exp(copyt:mul(self.beta)))
            end
            -- Divide by N
            s = s/self.kW
            -- log the summation of the exponents and multiply by inverse of beta
            s = torch.log(s)/self.beta
            -- copy to output
            output[{ {iter},{} }] = s
            iter = iter + 1
         end
      end

   -- if the input tensor is 3D (nBatchFrame x nInputFrame x inputFrameSize)
   else

      -- nOutputFrame
      nOutputFrame = (input:size(2) - self.kW)/self.dW + 1
      -- Output tensor
      output = torch.Tensor(input:size(1), nOutputFrame, input:size(3)):fill(0)
      -- perform log exponential pooling
      iter = 1 --to keep track of what frame we are updating in output
      for i=1,input:size(2),self.dW do
         -- will store the summation of the exponents
         s = torch.Tensor(input:size(1),1,input:size(3)):fill(0)
         -- calculate the summation of the exponents and store in s
         if (i+self.kW-1) <= input:size(2) then --if what the kernel envelopes is not outside the limit
            for j=1,i+self.kW-1 do
               -- create a copy of the input so we won't modify the input values
               copyt = torch.Tensor(input[{ {},{j},{} }]:size()):copy(input[{ {},{j},{} }])
               s:add(torch.exp(copyt:mul(self.beta)))
            end
            -- Divide by N
            s = s/self.kW
            -- log the summation of the exponents and multiply by inverse of beta
            s = torch.log(s)/self.beta
            -- copy to output
            output[{ {}, {iter}, {} }] = s
            iter = iter + 1
         end
      end

   end

   self.output = torch.Tensor(output:size()):copy(output)
   -----------------------------------------------
   return self.output
end

function TemporalLogExPooling:updateGradInput(input, gradOutput)

   -----------------------------------------------
   if input:dim() ==  2 then --2D

      gradInput = torch.Tensor(input:size()):fill(0)

      for i=1,input:size(2) do
         -- will store the gradient for current iteration. First copy the values of the frame multiplied by beta and take log
         grad = torch.Tensor(input[{ {},{i} }]:size()):copy(input[{ {},{i} }])
         grad = torch.exp( grad * self.beta )
         -- calculate sum
         for j= 1,gradOutput:size(1) do--for each value in gradOutput[j,{}], we should distribute the value to grdInput

            z = gradOutput[{j,i}]
            sum_of_exp_xk = grad[{  { (j-1)*self.dW + 1 , (j-1)*self.dW + self.kW }  }]:sum()

            for k=(j-1)*self.dW + 1,(j-1)*self.dW + self.kW do
               gradInput[{k,i}] = gradInput[{k,i}] + z * math.exp(self.beta * input[{k,i}]) / sum_of_exp_xk
            end
         
         end
      end

   else --3D
      --print("3D logexp")
      nBatchFrame = input:size(1)
      gradInput = torch.Tensor(input:size()):fill(0)

      for h=1,nBatchFrame do

         for i=1,input:size(3) do

            -- will store the gradient for current iteration. First copy the values of the frame multiplied by beta and take log
            grad = torch.Tensor(input[{ {h},{},{i} }]:size()):copy(input[{ {h},{},{i} }])
            grad = torch.exp( grad * self.beta )
            -- calculate sum
            for j= 1,gradOutput:size(2) do--for each value in gradOutput[j,{}], we should distribute the value to grdInput

               z = gradOutput[{h,j,i}]
               sum_of_exp_xk = grad[{  {1},{ (j-1)*self.dW + 1 , (j-1)*self.dW + self.kW },{1} }]:sum()

               for k=(j-1)*self.dW + 1,(j-1)*self.dW + self.kW do
                  gradInput[{h,k,i}] = gradInput[{h,k,i}] + z * math.exp(self.beta * input[{h,k,i}]) / sum_of_exp_xk
               end
            
            end
         end

      end

   end
   self.gradInput = torch.Tensor(gradInput:size()):copy(gradInput)
   -----------------------------------------------
   return self.gradInput
end

function TemporalLogExPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end


--function main()
--
--   model = nn.Sequential()
--   model = model:add(nn.TemporalLogExPooling(3,1,1))
--
--  x = torch.Tensor(3,4,5)
--   s = x:storage()
--   for i=1,s:size() do -- fill up the Storage
--     s[i] = i
--   end
--
--   ou = model:forward(x)
--   ou2 = model:backward(x,ou)
--end
