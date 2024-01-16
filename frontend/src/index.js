import React from 'react';
import ReactDOM from 'react-dom/client';
// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
import { RouterProvider } from 'react-router-dom';
import { router } from './router';
import { AuthProvider } from './contexts/AuthContext';
// Bootstrap Bundle JS
// import "bootstrap/dist/js/bootstrap.bundle.min";

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  </React.StrictMode>
);

