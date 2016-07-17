agent = {}

require 'nn'
require 'nngraph'

require 'DataBuffer'

local Agent = torch.class('agent.Agent')

function Agent:__init(model_file)
    torch.setdefaulttensortype('torch.FloatTensor')

    -- init agent parameters
    self.buf_len = torch.LongStorage{1000000}
    self.dims = {120,120}
    self.sample_len = 10
    self.batch_size = 50

    self.discount = 0.99
    self.lr = 0.01
    self.epsilon_start = 0.2
    self.epsilon_end = 0.01
    self.epsilon_step = 0.0001
    self.epsilon = self.epsilon_start

    self.nactions = 12


    self.start_training_after = self.batch_size + self.sample_len
    self.step = 0

    if model_file ~= nil then
        self.q_network = torch.load(model_file)
    else
        self.q_network = self:init_qnetwork()
    end
    self.target_network = self.q_network:clone()
    self.dataBuffer = agent.DataBuffer(self.bufLen, self.dims, self.sample_len)

    self.w, self.g = self.q_network:getParameters()
    self.gacc = self.g:clone():zero()
    self.tmp = self.g:clone()
end


function Agent:init_qnetwork()
    idat_img = nn.Identity()()
    ff = nn.SpatialConvolution(self.sample_len, 32, 3, 3)(idat_img)
    ff = nn.SpatialBatchNormalization(32)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialMaxPooling(2,2,2,2)(ff)

    ff = nn.SpatialConvolution(32, 32, 3, 3)(ff)
    ff = nn.SpatialBatchNormalization(32)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialConvolution(32, 48, 3, 3)(ff)
    ff = nn.SpatialBatchNormalization(48)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialMaxPooling(2,2,2,2)(ff)

    ff = nn.SpatialConvolution(48, 64, 3, 3)(ff)
    ff = nn.SpatialBatchNormalization(64)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialConvolution(64, 96, 3, 3)(ff)
    ff = nn.SpatialBatchNormalization(96)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialMaxPooling(2,2,2,2)(ff)

    ff = nn.SpatialConvolution(96, 128, 3, 3)(ff)
    ff = nn.SpatialBatchNormalization(128)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.SpatialMaxPooling(9,9,9,9)(ff)

    ff = nn.Reshape(128)(ff)

    ff = nn.Linear(128,self.nactions)(ff)

    ff = nn.gModule({idat_img}, {ff})

    return ff
end


function Agent:train(r,t,s)
    local a

    if self.step > self.start_training_after or t == 1 then
        a = self:egreedy(s)
    else
        a = torch.random(self.nactions)
    end

    self.dataBuffer:put(a,r,t,s)
    if self.step > self.start_training_after then
        self:run_minibatch()
    end
    self.step = self.step + 1
    return a
end


function Agent:run_minibatch()
    local s1,a,r,s2,t
    local q2, y, q, err
    local ai, err_selected

    s1,a,r,s2,t = self.dataBuffer:sample(self.batch_size)
    s1 = s1:float()
    s2 = s2:float()
    t = t:float()
    y = r:clone():float()

    if self.gpu then
        s1 = s1:cuda()
        s2 = s2:cuda()
        t = t:cuda()
        y = y:cuda()
    end

    q2 = self.target_network:forward(s2)
    -- add changes here for ddqn support
    q2_max,_ = q2:max(2)
    q2_max = q2_max:reshape(self.batch_size)
    q2_max:cmul(1-t)
    y:add(self.discount, q2_max)

    ai = a:reshape(agent_.batch_size,1):long()
    q1 = self.q_network:forward(s1)
    q1_selected = q1:gather(2,ai)

    err_selected = q1_selected - y

    err = q1:clone():zero()
    err:scatter(2,ai,err_selected)

    self.g:zero()
    self.q_network:backward(s1, err)

    self.tmp = torch.cmul(self.g, self.g)
    self.gacc:mul(0.95):add(0.05, self.tmp)

    self.tmp = torch.sqrt(self.gacc + 0.00001)
    self.g:cdiv(self.tmp)

    self.w:add(-self.lr, self.g)

    return err
end


function Agent:egreedy(s)
    local a
    if math.random() > self.epsilon then
        a = self:greedy(s)
    else
        a = torch.random(self.nactions)
    end
    return a
end


function Agent:greedy(s)
    local s2, q, a, _
    s2 = self.dataBuffer:tail(s):float()
    q = self.q_network:forward(s2)
    _,a = q:max(2)
    return a
end
