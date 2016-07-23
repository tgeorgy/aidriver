torch.setdefaulttensortype('torch.FloatTensor')

local app = require('waffle')
require 'json'

require 'util'
require 'Agent'

agent_instance = agent.Agent()
dims = {120,120}

app.post('/', function(req, res)
    r,t,s = process_input(req.body, dims)
    action = agent_instance:train(r,t,s)
    res.json({action=action-1})
end)

app.listen({port=5010})