import { Navigate, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import SinglePlayerPage from './pages/Single';
import TrainSetup from './pages/TrainSetup';
import Train from './pages/Train';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/single" element={<SinglePlayerPage />} />
      <Route path="/train-setup" element={<TrainSetup />} />
      <Route path="/train" element={<Train />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
