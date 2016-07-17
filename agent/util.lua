local read_terminate = function(istr)
    local s,mf,odat

    s = torch.CharStorage(#istr+1)
    s[#s] = 0
    mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    odat = mf:readByte()
    mf:close()

    return odat
end

local read_reward = function(istr)
    local s,mf,odat

    s = torch.CharStorage(#istr+1)
    s[#s] = 0
    mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    odat = mf:readInt()
    mf:close()

    return odat
end

local read_state = function(istr)
    local s,mf,odat

    s = torch.CharStorage(#istr+1)
    s[#s] = 0
    mf = torch.MemoryFile(s, 'rw')
    mf:writeString(istr)
    mf:seek(1):binary()
    odat = mf:readByte(#istr)
    mf:close()

    return odat
end

function process_input(istr, dims)
    local terminate, reward, state
    local state_byte_storage, pre_state

    terminate = read_terminate(string.sub(istr,1,1))
    reward = read_reward(string.sub(istr,2,5))
    state_byte_storage = read_state(string.sub(istr,6,#istr))
    state = torch.ByteTensor(state_byte_storage, 1, torch.LongStorage(dims))

    return terminate, reward, state
end
