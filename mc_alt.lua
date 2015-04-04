require 'torch'
require 'nn'
require 'optim'
require 'string'
require 'xlua'
require 'cunn'

ffi = require('ffi')

-- set seed for recreating tests
torch.manualSeed(1)

-- function to read in raw document data and convert to quantized vectors
function preprocess_data(raw_data, opt, dictionary)
    
    -- create empty tensors that will hold quantized data and labels
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.frame, opt.length)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]

            -- for keeping track of progress
            if ((i-1)*(opt.nTrainDocs+opt.nTestDocs) + j)%10000 == 0 then
            	print("Quantizing document " .. ((i-1)*(opt.nTrainDocs+opt.nTestDocs) + j))
            end

            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            -- create empty tensor to hold quantized text
            local q = torch.Tensor(opt.frame,opt.length):fill(0)

            -- will either scan the entire document or only go as far as length permits
            for c = 1,math.min(document:len(),opt.length) do
		        if dictionary[document:sub(c,c)] then
		        	q[dictionary[document:sub(c,c)]][c] = 1
		        end
		    end

            data[k] = q
            labels[k] = i
        end
    end

    return data, labels
end


function train_model(model, criterion, training_data, training_labels, opt)

	-- classes
	classes = {'1','2','3','4','5'}

	-- This matrix records the current confusion across classes
	confusion = optim.ConfusionMatrix(classes)

    parameters,gradParameters = model:getParameters()

    -- configure optimizer
    optimState = {
    	learningRate = opt.learningRate,
    	weightDecay = opt.weightDecay,
    	momentum = opt.momentum,
    	learningRateDecay = opt.learningRateDecay
    }
    optimMethod = optim.sgd

    epoch = epoch or 1
	local time = sys.clock()

	model:training()

	inputs = torch.zeros(opt.batchSize,opt.frame,opt.length):cuda()
	targets = torch.zeros(opt.batchSize):cuda()

	-- do one epoch
	print("\n==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,training_data:size(1),opt.batchSize do
		-- disp progress
		-- xlua.progress(t, training_data:size(1))
		inputs:zero()
		targets:zero()

		-- create mini batch
		if t + opt.batchSize-1 <= training_data:size(1) then
			-- xx = opt.batchSize
			inputs[{}] = training_data[{ {t,t+opt.batchSize-1},{},{} }]
			targets[{}] = training_labels[{ {t,t+opt.batchSize-1} }]
		
			-- create closure to evaluate f(X) and df/dX
			local feval = function(x)
				-- get new parameters
				if x ~= parameters then
					parameters:copy(x)
				end
				-- reset gradients
				gradParameters:zero()
				-- f is the average of all criterions
				local f = 0
				-- evaluate function for complete mini batch
				-- estimate f
				local output = model:forward(inputs:transpose(2,3))
				local err = criterion:forward(output, targets)
				f = f + err
				-- estimate df/dW
				local df_do = criterion:backward(output, targets)
				model:backward(inputs:transpose(2,3), df_do)
				-- update confusion
				for k=1,opt.batchSize do
					confusion:add(output[k], targets[k])
				end
				-- return f and df/dX
				return f,gradParameters
			end

			-- optimize on current mini-batch
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / training_data:size(1)
	print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	confusion:updateValids()

	-- print accuracy
	print("==> training accuracy for epoch " .. epoch .. ':')
	-- print(confusion)
	print(confusion.totalValid*100)

	-- save/log current net
	local filename = paths.concat(opt.save, 'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to '..filename)
	torch.save(filename, model)

	-- next epoch
	confusion:zero()
	epoch = epoch + 1

end

-- training function as it appears in the baseline code
-- function train_model(model, criterion, data, labels, test_data, test_labels, opt)

-- 	-- classes
-- 	classes = {'1','2','3','4','5'}

-- 	-- This matrix records the current confusion across classes
-- 	confusion = optim.ConfusionMatrix(classes)

--     parameters, grad_parameters = model:getParameters()

--     minibatch = torch.zeros(opt.batchSize, opt.frame, opt.length):cuda()
--     minibatch_labels = torch.zeros(opt.batchSize):cuda()
    
--     -- optimization functional to train the model with torch's optim library
--     local function feval(x) 
        
--         minibatch:zero()
--         minibatch_labels:zero()
--         minibatch[{}] = data[{ {opt.idx,opt.idx+opt.batchSize-1},{},{} }]
--         minibatch_labels[{}] = labels[{ {opt.idx,opt.idx+opt.batchSize-1} }]
        
--         model:training()
--         local minibatch_loss = criterion:forward(model:forward(minibatch:transpose(2,3):contiguous()), minibatch_labels)
--         model:zeroGradParameters()
--         model:backward(minibatch:transpose(2,3):contiguous(), criterion:backward(model.output, minibatch_labels))
        
--         return minibatch_loss, grad_parameters
--     end
    
--     for epoch=1,opt.nEpochs do
--         local order = torch.randperm(opt.nBatches) -- not really good randomization
--         for batch=1,opt.nBatches do
--             opt.idx = (order[batch] - 1) * opt.batchSize + 1
--             optim.sgd(feval, parameters, opt)
--             -- print("epoch: ", epoch, " batch: ", batch)
--         end

--         -- print("==> training accuracy for epoch " .. epoch .. "")
--         -- test_model(model, training_data, training_labels, opt)
--         print("==> testing on test set for epoch " .. epoch .. "")
--         test_model(model, test_data, test_labels, opt)
--         -- local accuracy = test_model(model, test_data, test_labels, opt)
--         -- print("epoch ", epoch, " error: ", accuracy)

--     end
-- end

function test_model(model, data, labels, opt)

	model:evaluate()

	t_input = torch.zeros(opt.frame, opt.length):cuda()
	t_labels = torch.zeros(1):cuda()
	-- test over test data
	for t = 1,data:size(1) do
		t_input:zero()
		t_labels:zero()
		t_input[{}] = data[t]
		t_labels[{}] = labels[t]
		local pred = model:forward(t_input:transpose(1,2))
		confusion:add(pred, t_labels[1])
	end
	-- print(confusion)
    confusion:updateValids()

    -- print accuracy
    -- print("==> test accuracy for epoch " .. epoch .. ':')
    -- print(confusion)
    print(confusion.totalValid*100)
	confusion:zero()
    
    -- EVALUATING THE MODEL IN THE BELOW FASHION CAUSES MEMORY ERRORS FOR THE GPU
    -- model:evaluate()

    -- t_data = torch.zeros(data:size()):cuda()
    -- t_labels = torch.zeros(labels:size()):cuda()

    -- t_data[{}] = data
    -- t_labels[{}] = labels

    -- local pred = model:forward(t_data:transpose(2,3):contiguous())
    -- local _, argmax = pred:max(2)
    -- local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    -- --local debugger = require('fb.debugger')
    -- --debugger.enter()

    -- return err
end


function main()

    -- define the character dictionary
    local alphabet =  "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    local dictionary = {}
    for i = 1,#alphabet do
    	dictionary[alphabet:sub(i,i)] = i
    end

    -- Configuration parameters
    opt = {}

    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- path to save model to
    opt.save = "results"
    -- number of frames in the character vectors
    opt.frame = alphabet:len() 
    -- maximum character size of text document
    opt.length = 1014
    -- training/test sizes
    opt.nTrainDocs = 10000
    opt.nTestDocs = 1000
    opt.nClasses = 5

    -- training parameters
    opt.nEpochs = 40
    opt.batchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.batchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 1e-5
    opt.momentum = 0.9
    opt.weightDecay = 0
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    processed_data, labels = preprocess_data(raw_data, opt, dictionary)

    print("Splitting data into training and validation sets...")
    -- split data into makeshift training and validation sets
    training_data = processed_data[{ {1,opt.nClasses*opt.nTrainDocs},{},{} }]:clone()
    training_labels = labels[{ {1,opt.nClasses*opt.nTrainDocs} }]:clone()
   
    -- test_data = training_data:clone()
    -- test_labels = training_labels:clone()

    if opt.nTestDocs > 0 then
	    test_data = processed_data[{ {(opt.nClasses*opt.nTrainDocs)+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)},{},{} }]:clone()
	    test_labels = labels[{ {(opt.nClasses*opt.nTrainDocs)+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)} }]:clone()
    else
        test_data = training_data:clone()
        test_labels = training_labels:clone()
	end

    -- build model *****************************************************************************
    model = nn.Sequential()
    -- first layer (#alphabet x 1014)
    model:add(nn.TemporalConvolution(opt.frame, 256, 7))
    model:add(nn.Threshold())
    model:add(nn.TemporalMaxPooling(3,3))

    -- second layer (336x256) 336 = (1014 - 7 / 1 + 1) / 3
    -- model:add(nn.TemporalConvolution(256, 256, 7))
    -- model:add(nn.Threshold())
    -- model:add(nn.TemporalMaxPooling(3,3))

    -- 1st fully connected layer (110x256) 110 = (330 - 7 / 1 + 1) / 3
    model:add(nn.Reshape(336*256))
    model:add(nn.Linear(336*256,1024))
    model:add(nn.Threshold())
    model:add(nn.Dropout(0.5))

    -- 2nd fully connected layer (1024)
    -- model:add(nn.Linear(1024,1024))
    -- model:add(nn.Threshold())
    -- model:add(nn.Dropout(0.5))

    -- final layer for classification
    model:add(nn.Linear(1024,5))
    model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()

	model:cuda()
	criterion:cuda()

	-- train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    -- local results = test_model(model, test_data, test_labels)
    -- print(results)
	-- print("\nTraining model...")
    for i=1,opt.nEpochs do
		train_model(model, criterion, training_data, training_labels, opt)
        test_model(model,test_data,test_labels,opt)
	end
    
end

main()