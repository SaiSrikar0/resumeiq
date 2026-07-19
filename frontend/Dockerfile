FROM node:20-alpine

WORKDIR /app

# Build context is frontend/ — package files are at the root of the context
COPY package*.json ./
RUN npm install

# Copy the rest of the frontend source code
COPY . .

# Build the application
RUN npm run build

EXPOSE 3000

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "3000"]
