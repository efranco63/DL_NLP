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
    

    data = torch.zeros(375036,opt.inputDim)
    count = 1
    for k,v in pairs(glove_table) do
       data[{ {count},{} }] = glove_table[k]:transpose(1,2)
       count = count + 1
    end

    -- calculate clusters
    print('Calculating kmeans clusters')
    clusters,clust_counts = unsup.kmeans(data,opt.clusters)

    print('Allocating cluster to each word')
    -- iterate through the words and find its closest cluster
    cluster_table = {}
    count = 1
    min_d = 1e10
    clust = 0
    for k,v in pairs(glove_table) do
        for j=1,opt.clusters do
            dist = torch.dist(data[count],clusters[{ {j},{} }])
            if dist < min_d then
                clust = j
                min_d = dist
            end
        end
        cluster_table[k] = clust
        count = count + 1
    end

    return cluster_table, clusters, clust_counts

end

function main()

    -- Configuration parameters
    opt = {}
    -- word vector dimensionality
    opt.inputDim = 300
    -- paths to glovee vectors and raw data
    opt.glovePath = "/home/eduardo/A3/glove/glove.6B." .. opt.inputDim .. "d.txt"
    -- path to save model to
    opt.save = "results"
    -- maximum number of words per text document
    opt.clusters = 2000


    print("Loading word vectors...")
    glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Computing kmeans clusters...")
    cluster_table, clusters, count = kmeans(glove_table)

    -- write table to file to import elsewhere
    file = torch.DiskFile('clusters_table.asc', 'w')
    file:writeObject(cluster_table)
    file:close()

end

main()