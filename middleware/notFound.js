function notFoundHandler(req, res) {
  res.status(404).render('error', {
    title: 'Page not found',
    message: 'The requested resource could not be located.'
  });
}

module.exports = { notFoundHandler };
