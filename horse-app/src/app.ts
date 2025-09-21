Sure, here's the proposed content for the `src/app.ts` file:

import express from 'express';
import horseRoutes from './routes/horseRoutes';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Routes
app.use('/api/horses', horseRoutes);

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});