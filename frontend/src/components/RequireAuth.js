import React from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
const RequireAuth = ({ children }) => {
    const { currentUser } = useAuth()
    console.log(currentUser)
    return (
        currentUser ? children : <Navigate to="/login" />
    )
}

export default RequireAuth