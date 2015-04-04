----------------------------------------------------------------------
-- This script produces a file 'predictions.csv' from an imported model 'model.net'
-- Will use the imported model to transform data which will be used to train the SVM
----------------------------------------------------------------------

require 'torch';
require 'xlua';
require 'nn';
require 'cunn';
require 'svm';

print '==> loading data'

-- load the model--------------------

print '==> loading Ed modexl'
--filename = 'model_1000.net'
--filename = 'blackbox.net'
--filename = 'blackbox_100.net'
filename = 'blackbox_2000.net'
model = torch.load(filename)
model:evaluate()
print '==> load Ed model completed...'


-------------prepare svm training data format

d_train= {};

ind = {};
for t=1,1600 do
    ind[t] = t;
end
ind = torch.IntTensor(ind);

for t=1,5000 do
  -- disp progress
  xlua.progress(t, train:size())
  -- get new sample
  local input = train.data[t]
  input = input:cuda()
  local pred = model:forward(input)
  local target = train.labels[t]
  table.insert( d_train , {target,{ind,pred:float()}} )     
end

--print(d)
LIBLINEAR_MODEL = liblinear.train(d_train);
labels,accuracy,dec = liblinear.predict(d_train,LIBLINEAR_MODEL);
LIBSVM_MODEL = libsvm.train(d_train);
labels,accuracy,dec = libsvm.predict(d_train,LIBSVM_MODEL);
--torch.save('SVM_LINEAR_MODEL.net', SVM_MODEL);
-- test training accurancy

--------------------------------------------------------------------

d_test= {};

ind = {};
for t=1,1600 do
    ind[t] = t;
end
ind = torch.IntTensor(ind);

for t=1,8000 do
  -- disp progress
  xlua.progress(t, test:size())
  -- get new sample
  local input = test.data[t]
  input = input:cuda()
  local pred = model:forward(input)
  local target = test.labels[t]
  table.insert( d_test , {target,{ind,pred:float()}} )     
end

labels,accuracy,dec = liblinear.predict(d_test,LIBLINEAR_MODEL);
labels,accuracy,dec = libsvm.predict(d_test,LIBSVM_MODEL);



