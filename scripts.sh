# Install Ruby (if not already installed)
sudo apt-get update
sudo apt-get install ruby-full

# Install Bundler
gem install bundler

# Install project dependencies
bundle install

# Build the static site (generates files in _site directory)
bundle exec jekyll build

# # Build with incremental updates (faster for development)
# bundle exec jekyll build --incremental

# # Start development server with live reload
# bundle exec jekyll serve

# # Start server accessible from any IP address
# bundle exec jekyll serve --host 0.0.0.0

# Start server on specific port
bundle exec jekyll serve --host 0.0.0.0 --port 4007

# # Start with incremental builds (faster)
# bundle exec jekyll serve --incremental

# # Start with live reload and watch for changes
# bundle exec jekyll serve --watch