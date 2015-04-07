require 'torch'
require 'nn'
require 'optim'
require 'unsup';

ffi = require('ffi')

-- set seed for recreating tests
torch.manualSeed(123)

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

function kmeans(glove_table)
    
    

    count = 1
    for k,v in pairs(glove_table) do
       data[{ {count},{} }] = glove_table[k]:transpose(1,2)
       count = count + 1
    end

    -- calculate clusters
    clusters,count = unsup.kmeans(data,500)

    dist = torch.dist(glove_table['dog'],clusters[{ {1},{} }])

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
    opt.glovePath = "/home/eduardo/A3/glove/glove.6B." .. opt.inputDim .. "d.txt"
    -- path to save model to
    opt.save = "results"
    -- maximum number of words per text document
    opt.length = 300
    -- training/test sizes
    opt.nTrainDocs = 24000
    opt.nTestDocs = 2000
    opt.nClasses = 5

    -- training parameters
    opt.nEpochs = 100
    opt.batchSize = 128
    opt.learningRate = 0.01
    opt.learningRateDecay = 1e-5
    opt.momentum = 0.9
    opt.weightDecay = 0

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    -- clusters_table = kmeans(glove_table)

end

main()