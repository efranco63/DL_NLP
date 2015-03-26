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

function TemporalLogExPooling:updateOutput(input)
   -----------------------------------------------
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
   self.output = torch.Tensor(output:size()):copy(output)
   -----------------------------------------------
   return self.output
end

function TemporalLogExPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   gradInput = torch.Tensor(input:size())
   for i=1,input:size(1) do
      -- will store the gradient for current iteration. First copy the values of the frame multiplied by beta and take log
      grad = torch.Tensor(input[{ {i},{} }]:size()):copy(input[{ {i},{} }])
      grad = torch.exp(grad*self.beta)
      -- will store the sum of the denominator and used in calculating grad
      sum = torch.Tensor(grad:size())
      -- calculate sum
      for j=1,input:size(1) do
         -- create a copy of the input frame so we won't modify the input values
         copyt = torch.Tensor(input[{ {j},{} }]:size()):copy(input[{ {j},{} }])
         sum:add(torch.exp(copyt:mul(self.beta)))
      end
      grad:cdiv(sum)
      -- assign to corresponding frame in gradInput
      gradInput[{ {i},{} }] = grad
   end

   -- SOMEHOW MULTIPLY BY GRADOUTPUT
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