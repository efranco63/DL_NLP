require 'torch'
require 'nn'
require 'optim'
require 'string'
require 'xlua'
require 'cunn'

ffi = require('ffi')

-- set seed for recreating tests
torch.manualSeed(8)

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

-- function to read in raw document data and convert to quantized vectors
function preprocess_train_data(raw_data, wordvector_table, opt)
    
    -- create empty tensors that will hold wordvector concatenations
    local data = torch.zeros(opt.nTrainDocs, opt.length, opt.inputDim)
    local labels = torch.zeros(opt.nTrainDocs)
    
    for j=1,opt.nTrainDocs do

        local index = raw_data.index[opt.idx][j]
        -- standardize to all lowercase
        local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
        local wordcount = 1
        -- break each review into words and concatenate into a vector thats of size length x inputDim
        for word in document:gmatch("%S+") do
            if wordcount < opt.length then
                if wordvector_table[word:gsub("%p+", "")] then
                    data[{ {j},{wordcount},{} }] = wordvector_table[word:gsub("%p+", "")]
                    wordcount = wordcount + 1
                end
            end
        end

        labels[j] = raw_data.labels[opt.idx][j]
    end

    return data, labels
end

function preprocess_test_data(raw_data, wordvector_table, opt)
    
    -- create empty tensors that will hold wordvector concatenations
    local data = torch.zeros(5*opt.nTestDocs, opt.length, opt.inputDim)
    local labels = torch.zeros(5*opt.nTestDocs)
    
    local counter = 1
    for i = 1,5 do
        for j=opt.nTrainDocs+1,opt.nTrainDocs+opt.nTestDocs do

            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            local wordcount = 1
            -- break each review into words and concatenate into a vector thats of size length x inputDim
            for word in document:gmatch("%S+") do
                if wordcount < opt.length then
                    if wordvector_table[word:gsub("%p+", "")] then
                        data[{ {counter},{wordcount},{} }] = wordvector_table[word:gsub("%p+", "")]
                        wordcount = wordcount + 1
                    end
                end
            end

            labels[counter] = raw_data.labels[i][j]
            counter = counter + 1
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

	inputs = torch.zeros(opt.batchSize,opt.length,opt.inputDim):cuda()
    -- inputs = torch.zeros(opt.batchSize,opt.length+4,opt.inputDim):cuda()
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
			local function feval(x) 
                
                local f = criterion:forward(model:forward(inputs), targets)
                model:zeroGradParameters()
                model:backward(inputs, criterion:backward(model.output, targets))

                for k=1,opt.batchSize do
                    confusion:add(model.output[k], targets[k])
                end
                
                return f, gradParameters
            end

			-- optimize on current mini-batch
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / training_data:size(1)
	print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	-- print(confusion)
	confusion:updateValids()

	-- print accuracy
	print("==> training accuracy for epoch " .. epoch .. ':')
	accuracy = confusion.totalValid*100
    -- log accuracy for this epoch
    print(accuracy)

end

function test_model(model, data, labels, opt)

    model:evaluate()

    t_input = torch.zeros(opt.length, opt.inputDim):cuda()
    -- t_input = torch.zeros(opt.length+4, opt.inputDim):cuda()
    t_labels = torch.zeros(1):cuda()
    -- test over test data
    for t = 1,data:size(1) do
        t_input:zero()
        t_labels:zero()
        t_input[{}] = data[t]
        t_labels[{}] = labels[t]
        local pred = model:forward(t_input)
        confusion:add(pred, t_labels[1])
    end
    -- print(confusion)
    confusion:updateValids()

    -- print accuracy
    print("==> test accuracy for epoch " .. epoch .. ':')
    -- print(confusion)
    accuracy = confusion.totalValid*100
    print(accuracy)

    -- save/log current net
    if accuracy > accs['max'] then 
        local filename = paths.concat(opt.save, 'model13.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end

    -- if accuracy <= accs['max'] then
    --     opt.learningRate = opt.learningRate/10
    -- end

    accs['max'] = math.max(accuracy,accs['max'])
    accs[epoch] = accuracy

    confusion:zero()
end


function main()

    -- Configuration parameters
    opt = {}
    -- table acting as a log of accuracies per epoch
    accs = {}
    accs['max'] = 0
    -- word vector dimensionality
    opt.inputDim = 300
    -- paths to glovee vectors and raw data
    opt.glovePath = "/scratch/courses/DSGA1008/A3/glove/glove.6B." .. opt.inputDim .. "d.txt"
    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- path to save model to
    opt.save = "results"
    -- maximum number of words per text document
    opt.length = 200
    -- training/test sizes per class
    opt.nTrainDocs = 97500
    opt.nTestDocs = 32500

    -- training parameters
    opt.init_weight = 0.1 -- random weight initialization
    opt.nEpochs = 50
    opt.batchSize = 64
    opt.learningRate = 0.01
    opt.learningRateDecay = 1e-5
    opt.momentum = 0.9
    opt.weightDecay = 0

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)

    -- shuffle all the data in a way in which we can call on it five times for conserving memory
    print("Shuffling data...")
    s_raw_data = {}
    s_raw_data['index'] = torch.reshape(raw_data.index,1,5*130000)
    s_raw_data['labels'] = torch.zeros(650000)
    for i=1,5 do
        s_raw_data['labels'][{ {130000*(i-1)+1,i*130000} }] = i
    end
    order = torch.randperm(650000)

    index_table = {}
    for i=1,5 do
        index_table[i] = order[{ {130000*(i-1)+1,i*130000} }]
    end

    labels = torch.zeros(5,130000)
    index_n = torch.zeros(5,130000)
    for j=1,5 do
        for i=1,130000 do
            index_n[{ {j},{i} }] = s_raw_data.index[{ {},{index_table[j][i]} }]
            labels[{ {j},{i} }] = s_raw_data.labels[index_table[j][i]]
        end
    end

    n_raw_data = {}
    n_raw_data['index'] = index_n:clone()
    n_raw_data['labels'] = labels:clone()
    n_raw_data['content'] = raw_data.content:clone()
    -----------------------------------------------------------------------------------

    -- build model *****************************************************************************
    model = nn.Sequential()
    -- first layer (#inputDim x 204)
    model:add(nn.TemporalConvolution(opt.inputDim, 1024, 7))
    model:add(nn.Threshold())
    model:add(nn.TemporalMaxPooling(2,2))

    -- second layer (147x512) 
    model:add(nn.TemporalConvolution(1024, 1024, 7))
    model:add(nn.Threshold())
    model:add(nn.TemporalMaxPooling(3,3))

    -- 1st fully connected layer (19x512)
    model:add(nn.Reshape(30*1024))
    model:add(nn.Linear(30*1024,1024))
    model:add(nn.Threshold())
    model:add(nn.Dropout(0.7))

    -- final layer for classification 1024
    model:add(nn.Linear(1024,5))
    model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()

	-- CUDA
	model:cuda()
	criterion:cuda()

    -- randomly initialize weights  
    -- model:getParameters():uniform(-opt.init_weight, opt.init_weight)

	print("\nTraining model...")
    for i=1,opt.nEpochs do
        epoch = i
        for j=1,5 do
            opt.idx = j
            print("Processing data batch " .. j .. " for epoch " .. i)
            local training_data, training_labels = preprocess_train_data(n_raw_data, glove_table, opt)
    		train_model(model, criterion, training_data, training_labels, opt)
        end
        confusion:zero()
        local test_data, test_labels = preprocess_test_data(n_raw_data, glove_table, opt)
        test_model(model,test_data,test_labels,opt)
	end

end

main()