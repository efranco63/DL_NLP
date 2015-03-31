require 'torch';
ffi = require 'ffi';
require 'string';

-- the dictionary of characters, 69
dictionary = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
train = torch.load("/scratch/courses/DSGA1008/A3/data/train.t7b")

function quantization(document,frame,length)
    document = document:lower()
    x = torch.Tensor(frame,length):fill(0)
    for i = 1,length do
        character = document:sub(i,i)
        position = string.find(dictionary,character)
        if position ~= nil then
            x[position][i] = 1 
        end
    end
    return x
end

function gettensor(i, j, frame, length)
    index = train.index[i][j]
    document = ffi.string(torch.data(train.content:narrow(1, index, 1)))
    -- x is input tensor [frame x length]
    x = quantization(document, frame, length)
    return x
end

----------------------------------
----------------------------------
-- i=2
-- j=100
-- frame = dictionary:len()
-- length = 256
-- x = gettensor(i, j, frame, length)
