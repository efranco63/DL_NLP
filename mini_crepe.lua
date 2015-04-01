require 'torch'
require 'nn'
require 'optim'
require 'string'

ffi = require('ffi')

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
    -- number of frames in the character vectors
    opt.frame = alphabet:len() 
    -- maximum character size of text document
    opt.length = 200
    -- training/test sizes
    opt.nTrainDocs = 20000
    opt.nTestDocs = 5000
    opt.nClasses = 5
    
    print("Loading raw data...")
    raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    processed_data, labels = preprocess_data(raw_data, opt, dictionary)

    print("Splitting data into training and validation sets...")
    -- split data into makeshift training and validation sets
    training_data = processed_data[{ {1,opt.nClasses*opt.nTrainDocs},{},{} }]:clone()
    training_labels = labels[{ {1,opt.nClasses*opt.nTrainDocs} }]:clone()
   
    test_data = processed_data[{ {(opt.nClasses*opt.nTrainDocs)+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)},{},{} }]:clone()
    test_labels = labels[{ {(opt.nClasses*opt.nTrainDocs)+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)} }]:clone()

    -- -- construct model:
    -- model = nn.Sequential()
   
    -- -- if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    -- model:add(nn.TemporalConvolution(1, 40, 10, 1))
    
    -- --------------------------------------------------------------------------------------
    -- -- Replace this temporal max-pooling module with your log-exponential pooling module:
    -- --------------------------------------------------------------------------------------
    -- model:add(nn.TemporalMaxPooling(3, 1))
    -- -- beta = 15
    -- -- model:add(nn.TemporalLogExpPooling(3, 1, beta))
    
    -- -- 20*39
    -- model:add(nn.Reshape(40*189, true))
    -- model:add(nn.Linear(40*189, 5))
    -- model:add(nn.LogSoftMax())

    -- criterion = nn.ClassNLLCriterion()
   
    -- train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    -- local results = test_model(model, test_data, test_labels)
    -- print(results)
end

main()