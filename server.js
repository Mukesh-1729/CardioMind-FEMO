const { createApp } = require('./app');
const { config } = require('./config/env');

const app = createApp();

app.listen(config.port, () => {
  console.log(`CardioMind web app listening on port ${config.port}`);
});
