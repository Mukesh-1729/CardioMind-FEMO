const express = require('express');
const path = require('path');
const helmet = require('helmet');
const compression = require('compression');
const cors = require('cors');
const expressLayouts = require('express-ejs-layouts');
const session = require('express-session');
const flash = require('connect-flash');
const morgan = require('morgan');

const { config } = require('./config/env');
const { configureViews } = require('./config/viewEngine');
const { initializeDatabase } = require('./config/database');
const { createSessionStore } = require('./config/sessionStore');
const { attachRoutes } = require('./routes');
const { notFoundHandler } = require('./middleware/notFound');
const { errorHandler } = require('./middleware/errorHandler');
const { syncCurrentUser } = require('./middleware/authMiddleware');

function createApp() {
  const app = express();

  if (config.nodeEnv === 'production') {
    app.set('trust proxy', 1);
  }

  initializeDatabase();
  configureViews(app);
  app.use(expressLayouts);
  app.set('layout', 'layouts/main');

  app.use(helmet());
  app.use(compression());
  app.use(cors({ origin: config.corsOrigin, credentials: true }));
  app.use(morgan(config.nodeEnv === 'production' ? 'combined' : 'dev'));
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));
  app.use('/public', express.static(path.join(__dirname, 'public')));
  app.use(
    session({
      name: config.sessionName,
      secret: config.sessionSecret,
      resave: false,
      saveUninitialized: false,
      rolling: true,
      store: createSessionStore(),
      cookie: {
        secure: config.nodeEnv === 'production',
        httpOnly: true,
        sameSite: 'lax',
        maxAge: config.sessionTtlMs
      }
    })
  );
  app.use(flash());
  app.use(syncCurrentUser);

  app.use((req, res, next) => {
    res.locals.appName = 'CardioMind';
    res.locals.currentPath = req.path;
    res.locals.user = req.session?.doctor || null;
    res.locals.currentUser = req.session?.doctor || null;
    res.locals.csrfToken = '';
    res.locals.flash = {
      success: req.flash('success'),
      error: req.flash('error')
    };
    next();
  });

  attachRoutes(app);
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

module.exports = { createApp };
