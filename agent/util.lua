local read_terminate = function(istr)
    local s = torch.CharStorage(#istr+1)
    s[#s] = 0
    local mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    local odat = mf:readByte()
    mf:close()
    return odat
end

local read_reward = function(istr)
    local s = torch.CharStorage(#istr+1)
    s[#s] = 0
    local mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    local odat = mf:readInt()
    mf:close()
    return odat
end

local read_state = function(istr)
    local s = torch.CharStorage(#istr+1)
    s[#s] = 0
    local mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    local odat = mf:readByte(#istr)
    mf:close()
    return odat
end

function process_input(istr, dims)
    local terminate = read_terminate(string.sub(istr,1,1))
    local reward = read_reward(string.sub(istr,2,5))
    local state_byte_storage = read_state(string.sub(istr,6,#istr))
    local pre_state = torch.ByteTensor(state_byte_storage, 1, dims)
    state = torch.ByteTensor(3,480,480):zero()
    state[1] = pre_state:eq(1)
    state[2] = pre_state:eq(2)
    state[3] = pre_state:eq(3)
    return terminate, reward, state
end
