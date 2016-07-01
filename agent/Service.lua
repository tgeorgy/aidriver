local app = require('waffle')
require 'json'

app.post('/', function(req, res)
    print(#req.body)
    res.json({action=6})
end)

app.listen({port=5010})