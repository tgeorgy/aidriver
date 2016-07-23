local DataBuffer = torch.class('agent.DataBuffer')


function DataBuffer:__init(buf_len, dims, sample_len)
    self.buf_len = buf_len or 1000000
    self.dims = dims or torch.LongTensor{480,480}
    self.sample_len = sample_len or 10

    self.buf_s = torch.ByteTensor(self.buf_len, self.dims[1], self.dims[2])
    self.buf_t = torch.ByteTensor(self.buf_len):fill(1)
    self.buf_r = torch.LongTensor(self.buf_len)
    self.buf_a = torch.ByteTensor(self.buf_len)
    self.buf_p = torch.FloatTensor(self.buf_len):zero()

    self.s1_addr = {}
    self.s1_addr_last = nil

    self.last_id = 0
end


function DataBuffer:put(a,r,t,s)
    local current_id, errase_id
    current_id = (self.last_id % self.buf_len) + 1

    self.buf_t[current_id] = t
    self.buf_r[current_id] = r
    self.buf_s[current_id] = s
    self.buf_a[current_id] = a
    self.buf_p[current_id] = 0.1

    self.s1_addr[#self.s1_addr + 1] = self.s1_addr_last

    if torch.isTensor(t) then t = t[1] end
    if t == 0 then
        self.s1_addr_last = current_id
    else
        self.s1_addr_last = nil
    end

    errase_id = current_id + self.sample_len - 2
    errase_id = (errase_id % self.buf_len) + 1

    if self.s1_addr[1] == errase_id then
        table.remove(self.s1_addr, 1)
    end
    self.last_id = current_id
end


function DataBuffer:tail(s)
    local tail, tail_id
    tail_id = self.last_id

    tail = torch.ByteTensor(1, self.sample_len, self.dims[1], self.dims[2])

    if self.buf_t[tail_id] == 1 then

        for i  = 1,self.sample_len do
            tail[{1,i}] = s
        end

    else

        tail[{1,1}] = s
        for i  = 2,self.sample_len do
            tail[{1,i}] = self.buf_s[tail_id]
            tail_id = (tail_id - 2 + self.buf_len) % self.buf_len + 1
            tail_id = tail_id + self.buf_t[tail_id]
        end

    end

    return tail
end


function DataBuffer:pop_tail(s)
    -- Get delayed frames for last state
    local tail

    tail = torch.ByteTensor(1, self.sample_len-1, self.dims[1], self.dims[2])
    for i  = 1,self.sample_len-1 do
        tail[{1,i}] = self.buf_s[tail_id]
        tail_id = (tail_id - 2 + self.buf_len) % self.buf_len + 1
        tail_id = tail_id + self.buf_t[tail_id]
    end
    return tail
end


function DataBuffer:update_p(err, s1_ids)
    -- Update priority distribution for sampled states
    self.buf_p:indexCopy(1, s1_ids, err)
end


function DataBuffer:sample(n, replacement, include_last)
    local s1_addr_t, probs
    local sample_ids, s1_ids, s2_ids
    local rem = torch.remainder

    include_last = include_last or true
    replacement = replacement or true

    assert(n < #self.s1_addr,
        'Too early - collect more data before training')
    assert(n < self.buf_len - self.sample_len + 1,
        'Too large batch size - increase buffer length')

    -- Prioritized sampling
    -- [http://arxiv.org/pdf/1511.05952v4.pdf]
    s1_addr_t = torch.LongTensor(self.s1_addr)
    probs = self.buf_p:index(1,s1_addr_t)
    -- without replacement only
    sample_ids = torch.multinomial(probs, n, false)
    s1_ids = s1_addr_t:index(1,sample_ids)
    --[[
    if replacement then
        sample_ids = torch.rand(n)
        sample_ids = sample_ids * (#self.s1_addr)
        sample_ids = torch.ceil(sample_ids)
    else
        sample_ids = torch.randperm(#self.s1_addr)
        sample_ids = sample_ids[{{1,n}}]
    end
    s1_ids = sample_ids:clone():long()

    for i = 1,n do
        s1_ids[i] = self.s1_addr[ sample_ids[i] ]
    end
    if include_last then
        s1_ids[1] = self.s1_addr[#self.s1_addr]
    end
    --]]
    s2_ids = (s1_ids:clone() % self.buf_len) + 1


    local s1,a,r,s2,t
    local t1,t2

    a = self.buf_a:index(1, s1_ids)
    r = self.buf_r:index(1, s2_ids)
    t = self.buf_t:index(1, s2_ids)

    -- restore states (multiple delayed frames per state)
    s1 = torch.ByteTensor(n, self.sample_len, self.dims[1], self.dims[2])
    s2 = torch.ByteTensor(n, self.sample_len, self.dims[1], self.dims[2])

    for i = 1,self.sample_len do
        s1[{{1,n},i}] = self.buf_s:index(1, s1_ids)
        s2[{{1,n},i}] = self.buf_s:index(1, s2_ids)

        -- error computing reminder for negative integer
        s1_ids = (s1_ids - 2 + self.buf_len) % self.buf_len + 1
        s2_ids = (s2_ids - 2 + self.buf_len) % self.buf_len + 1

        -- sample same frame again if it is first in iteration
        t1 = self.buf_t:index(1, s1_ids)
        t2 = self.buf_t:index(1, s2_ids)

        s1_ids = s1_ids % self.buf_len + t1:long()
        s2_ids = s2_ids % self.buf_len + t2:long()
    end

    return s1,a,r,s2,t,s1_ids
end

