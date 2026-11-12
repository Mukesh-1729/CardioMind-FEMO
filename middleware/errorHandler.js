function errorHandler(error, req, res, next) {
  const statusCode = error.statusCode || 500;
  console.error('SERVER ERROR [' + req.method + ' ' + req.url + ']:', error);

  if (res.headersSent) {
    return next(error);
  }

  // Ensure layout variables exist even if the error occurred before
  // the res.locals middleware had a chance to run (e.g. session store failure).
  if (!res.locals.appName) {
    res.locals.appName = 'CardioMind';
  }
  if (!res.locals.currentUser) {
    res.locals.currentUser = null;
  }
  if (!res.locals.flash) {
    res.locals.flash = { success: [], error: [] };
  }
  if (!res.locals.currentPath) {
    res.locals.currentPath = req.path || '/';
  }

  res.status(statusCode).render('error', {
    title: 'Application error',
    message: statusCode === 500 ? 'An unexpected error occurred.' : error.message
  });
}

module.exports = { errorHandler };
