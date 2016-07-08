torch.setdefaulttensortype('torch.FloatTensor')
local app = require('waffle')
require 'json'
require 'util'
require 'Agent'

agent.Agent()

app.post('/', function(req, res)
    t,r,s = process_input(#req.body)
    action = agent.train_iter(t,r,s)
    res.json({action=action})
end)

app.listen({port=5010})