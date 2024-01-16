import { createBrowserRouter } from "react-router-dom";
import App from "../pages/App";
import RegisterPage from "../pages/Register/RegisterPage";
import Pulse from "../pages/Pulse/Pulse";
import LoginPage from "../pages/Login/LoginPage";
import RequireAuth from "../components/RequireAuth";
import Deneme from "../pages/Deneme";
import MyTimer from "../pages/MyTimer";

export const router = createBrowserRouter([
    {
        path:"/",
        element: 
        <RequireAuth>
            <App/>
        </RequireAuth>
    },
    {
        path:"/register",
        element: <RegisterPage/>
    },
    {
        path:"/pulse",
        element: 
            <Pulse/>
    },
    {
        path:"/login",
        element: <LoginPage/>
    },
    {
        path:"/deneme",
        element: <Deneme/>
    },
    {
        path:"/timer",
        element: <MyTimer/>
    }
])