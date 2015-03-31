require 'torch';
ffi = require 'ffi';
require 'string';

-- the dictionary of characters, 69
dictionary = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
train = torch.load("/scratch/courses/DSGA1008/A3/data/train.t7b")

-- function to transform a text of words into 69 (len of dictionary) x length 1-of-m encoded tensor
function quantization(document,frame,length)
    -- convert to all lower case
    document = document:lower()
    -- make an empty tensor to hold the values
    x = torch.Tensor(frame,length):fill(0)
    -- will either scan the entire document or only go as far as length permits
    for i = 1,math.min(document:len(), length) do
        character = document:sub(i,i)
        position = string.find(dictionary,character)
        -- if the character is in the dictionary, add a 1 in its corresponding place in its vector
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
