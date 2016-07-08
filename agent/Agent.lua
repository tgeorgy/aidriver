require 'DataBuffer'

Agent = torch.class('Agent')

function Agent:__init(model_file)
    -- init agent parameters
    self.buf_len = torch.LongStorage{10000}
    self.dims = {120,120}
    self.sample_len = 10
    self.batch_size = 100

    self.discount = 0.99
    self.lr = 0.01
    self.epsilon_start = 0.2
    self.epsilon_end = 0.01
    self.epsilon_step = 0.0001
    self.epsilon = self.epsilon_start

    self.nactions = 12


    self.start_training_after = self.batch_size + self.sample_len
    self.step = 0

    if model_file != nil then
        self.q_network = torch.load(model_file)
    else
        self.q_network = self.init_qnetwork()
    end
    self.target_network = self.q_network:clone()
    self.dataBuffer = DataBuffer(self.bufLen, self.dims, self.sample_len)

    self.w, self.g = self.q_network:getParameters()
    self.gacc = self.g:clone():zero()
    self.tmp = self.g:clone()
end


function Agent:init_qnetwork()

end


function Agent:train(r,t,s)
    if  then
        a = 1
    elseif t == 1 or self.step <= self.sample_len then
        a = torch.random(self.nactions)
    else
        a = self:egreedy(s)
    end

    self.dataBuffer:put(a,r,t,s)
    if self.step > self.start_training_after then
        self:run_minibatch(sample)
    end
    self.step = self.step + 1
end


function Agent:run_minibatch(batch)
    a,r,t,s1,s2 = self.dataBuffer:sample(self.batch_size)
    q2 = self.target_network:forward(s2)
    y = r + self.discount*(1-t)*q2
    q = self.q_network:forward(s1)
    err = q-q2

    ai = a:reshape(self.batch_size,1):long()
    err_selected = err:gather(2,ai)
    err:zero():scatter(2,ai,err_selected)

    self.g:zero()
    self.q_network:backward(s1, err)

    self.tmp = torch.cmul(self.g, self.g)
    self.gacc:mull(0.95):add(0.05, self.tmp)

    self.tmp = torch.sqrt(self.gacc)
    self.g:cdiv(self.tmp)

    self.w:add(-self.lr, self.g)

    return err
end


function Agent:egreedy(s)
    if math.random() > self.epsilon then
        a = self:greedy(s)
    else
        a = torch.random(self.nactions)
    end
    return a
end


function Agent:greedy(s)
    s2 = self.dataBuffer:tail(s)
    q = self.q_network:forward(s2)
    _,a = q:max(2)
    return a
end