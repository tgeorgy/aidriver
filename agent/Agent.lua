agent = {}

require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'

require 'DataBuffer'

local Agent = torch.class('agent.Agent')

function Agent:__init(model_dir, arg)
    torch.setdefaulttensortype('torch.FloatTensor')

    -- init agent parameters
    arg = arg or {}
    self.buf_len    = arg.buf_len or 1000000
    self.dims       = arg.dims or {40,40}
    self.sample_len = arg.sample_len or 10
    self.batch_size = arg.batch_size or 32

    self.discount = arg.discount or 0.99
    self.lr       = arg.lr or 0.001
    self.epsilon_start = arg.epsilon_start or 0.5
    self.epsilon_end   = arg.epsilon_end or 0.1
    self.epsilon_end_t  = arg.epsilon_end_t or 500000

    self.clip_delta = 1
    self.scale_reward = 0.1

    self.nactions = arg.nactions or 12

    self.gpu = arg.gpu or true

    self.update_target_every  = arg.update_target_every or 10000
    self.save_agent_every     = arg.save_agent_every or 2000
    self.print_every          = arg.print_every or 10
    self.clean_every          = arg.clean_every or 100
    self.start_training_after = arg.start_training_after or self.batch_size + self.sample_len + 10
    self.change_target_after  = arg.change_target_after or 20000

    self.step = 0
    self.game = 0
    self.total_reward = 0

    self.model_dir = model_dir or './save'
    if not paths.dirp(self.model_dir) then
        paths.mkdir(self.model_dir)
    end

    self.agent_file = paths.concat(self.model_dir,'agent.t7')
    self.model_file = paths.concat(self.model_dir,'qnet.t7')

    if model_dir ~= nil then
        self.q_network = torch.load(self.model_file)
    else
        self.q_network = self:init_qnetwork()
    end

    if self.gpu then
        self.q_network = self.q_network:cuda()
        cudnn.convert(self.q_network, cudnn);
    end

    self.target_network = self.q_network:clone()
    self.dataBuffer = agent.DataBuffer(self.buf_len, self.dims, self.sample_len)

    self.w, self.g = self.q_network:getParameters()
    self.gacc = self.g:clone():zero()
    self.tmp = self.g:clone()

    self.log_file = self:init_log()
end


function Agent:init_qnetwork()
    idat_img = nn.Identity()()
    ff_l = nn.SpatialConvolution(self.sample_len, 16, 3, 3, 2, 2)(idat_img)
    ff_r = nn.SpatialConvolution(self.sample_len, 16, 9, 1, 2, 1, 3, 0)(idat_img)
    ff_r = nn.SpatialConvolution(16, 16, 1, 9, 1, 2, 0, 3)(ff_r)

    ff = nn.JoinTable(2)({ff_l,ff_r})
    ff = nn.SpatialBatchNormalization(32)(ff)
    ff = nn.ReLU()(ff)

    ff_l = nn.SpatialConvolution(32, 32, 3, 3, 2, 2)(ff)
    ff_r = nn.SpatialConvolution(32, 32, 9, 1, 2, 1, 3, 0)(ff)
    ff_r = nn.SpatialConvolution(32, 32, 1, 9, 1, 2, 0, 3)(ff_r)

    ff = nn.JoinTable(2)({ff_l,ff_r})
    ff = nn.SpatialBatchNormalization(64)(ff)
    ff = nn.ReLU()(ff)

    ff_l = nn.SpatialConvolution(64, 32, 3, 3)(ff)
    ff_r = nn.SpatialConvolution(64, 32, 9, 1, 1, 1, 3, 0)(ff)
    ff_r = nn.SpatialConvolution(32, 32, 1, 9, 1, 1, 0, 3)(ff_r)

    ff = nn.JoinTable(2)({ff_l,ff_r})
    ff = nn.SpatialBatchNormalization(64)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.Reshape(64*7*7)(ff)
    ff = nn.Linear(64*7*7,512)(ff)
    ff = nn.ReLU()(ff)

    ff = nn.Linear(512,self.nactions)(ff)

    ff = nn.gModule({idat_img}, {ff})

    -- init small values
    linear_modules = ff:findModules('nn.Linear')
    for i = 1,#linear_modules do
        linear_modules[i].weight:mul(0.7)
        linear_modules[i].bias:fill(0)
    end

    return ff
end


function Agent:train(r,t,s)
    local a, offset

    if self.step % self.clean_every == 1 then
        collectgarbage()
    end

    if self.step > self.start_training_after and t == 0 then
        a = self:egreedy(s)
    else
        a = torch.random(self.nactions)
    end

    self.total_reward = self.total_reward+r
    self.dataBuffer:put(a,r,t,s)
    offset = 0
    if self.step > self.start_training_after then
        offset = self:run_minibatch()
        if self.step % self.update_target_every == 1 and
                self.step > self.change_target_after then
            self.target_network = self.q_network:clone()
            self.dataBuffer.buf_p:fill(0.1)
        end
        if self.step % self.save_agent_every == 1 then
            --torch.save(self.agent_file, self)
            torch.save(self.model_file, self.q_network)
        end
    end

    if self.step % self.print_every == 1 or t == 1 then
        self.log_file:writeString('\n\nstep: ' .. self.step)
        self.log_file:writeString('\ngame: ' .. self.game)
        self.log_file:writeString('\noffset: ' .. offset)
        self.log_file:writeString('\nreward: ' .. self.total_reward)
        self.log_file:writeString('\nterm: ' .. t)
        if t == 1 then
            print('game ' .. self.game)
            print('reward ' .. self.total_reward)
        end
    end

    if t == 1 then
        self.game = self.game+1
        self.total_reward = 0
    end

    self.step = self.step + 1
    return a
end


function Agent:run_minibatch()
    local s1, a, r, s2, t
    local q2, y, q, err, q2_maxi, q2_max
    local ai, err_selected, prob_update, _, cost

    s1,a,r,s2,t,s1_ids = self.dataBuffer:sample(self.batch_size)

    s1 = s1:float()
    a = a:long()
    s2 = s2:float()
    t = t:float()
    y = r:clone():float():mul(self.scale_reward)

    if self.gpu then
        s1 = s1:cuda()
        a = a:cuda()
        s2 = s2:cuda()
        t = t:cuda()
        y = y:cuda()
    end

    -- Double DQL
    -- [http://arxiv.org/pdf/1509.06461v3.pdf]
    q2 = self.target_network:forward(s2/32)
    _,q2_maxi = self.q_network:forward(s2/32):max(2)
    q2_max = q2:gather(2,q2_maxi)
    --q2_max,_ = q2:max(2)
    q2_max = q2_max:reshape(self.batch_size)
    -- Commenting following out because there is nothing indicating
    -- terminate frame - the game just brakes
    -- q2_max:cmul(1-t) -- zero for terminate state
    y:add(self.discount, q2_max)

    ai = a:reshape(self.batch_size,1)
    q1 = self.q_network:forward(s1/32)
    q1_selected = q1:gather(2, ai)

    err_selected = q1_selected - y
    prob_update = err_selected:clone():float()
    prob_update = prob_update:abs():add(self.step/2000000)
    prob_update = prob_update:reshape(self.batch_size)
    self.dataBuffer:update_p(prob_update, s1_ids)

    err_selected[torch.gt(err_selected, self.clip_delta)] = self.clip_delta
    err_selected[torch.lt(err_selected, -self.clip_delta)] = -self.clip_delta

    cost = torch.pow(err_selected,2):mean()^0.5

    err = q1:zero()
    err:scatter(2, ai, err_selected)

    self.g:zero()
    self.q_network:backward(s1, err)

    -- rmsprop
    self.tmp = torch.cmul(self.g, self.g)
    self.gacc:mul(0.95):add(0.05, self.tmp)

    self.tmp = torch.sqrt(self.gacc + 0.00001)
    self.g:cdiv(self.tmp)

    self.w:add(-self.lr, self.g)

    return cost
end


function Agent:egreedy(s)
    local a, epsilon

    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)*
        math.max(0, (self.epsilon_end_t - self.step)/self.epsilon_end_t)

    if math.random() > epsilon then
        a = self:greedy(s)
    else
        a = torch.random(self.nactions)
    end

    return a
end


function Agent:greedy(s)
    local s2, q, a, _

    s2 = self.dataBuffer:tail(s):float()
    if self.gpu then
        s2 = s2:cuda()
    end
    q = self.q_network:forward(s2/32)
    _,a = q:max(2)

    return a:byte()[{1,1}]
end


function Agent:init_log()
    local acc, acc_s, log_file_name

    acc = 0
    for f in paths.files(self.model_dir) do
        if string.sub(f,#f-3,#f) == '.log' then
            acc = acc+1
        end
    end

    acc_s = string.format('%0.4f',acc/10000)
    acc_s = string.sub(acc_s, 3,6)
    log_file_name = paths.concat(self.model_dir, acc_s .. '.log')

    log_file = torch.DiskFile(log_file_name, 'w')
    log_file:noBuffer()

    log_file:writeString('PARAMETERS\n')
    log_file:writeString('\nbuf_len: ' .. self.buf_len)
    log_file:writeString('\ndims: ' .. self.dims[1] .. ' x ' .. self.dims[2])
    log_file:writeString('\nsample_len: ' .. self.sample_len)
    log_file:writeString('\nbatch_size: ' .. self.batch_size)
    log_file:writeString('\ndiscount: ' .. self.discount)
    log_file:writeString('\nlr: ' .. self.lr)
    log_file:writeString('\nepsilon_start: ' .. self.epsilon_start)
    log_file:writeString('\nepsilon_end: ' .. self.epsilon_end)
    log_file:writeString('\nepsilon_end_t: ' .. self.epsilon_end_t)
    log_file:writeString('\nnactions: ' .. self.nactions)
    log_file:writeString('\ngpu: ' .. tostring(self.gpu))
    log_file:writeString('\nupdate_target_every: ' .. self.update_target_every)
    log_file:writeString('\nsave_agent_every: ' .. self.save_agent_every)
    log_file:writeString('\nclean_every: ' .. self.clean_every)
    log_file:writeString('\nstart_training_after: ' .. self.start_training_after)

    log_file:writeString('\n\nTRAINING\n\n')

    return log_file
end
