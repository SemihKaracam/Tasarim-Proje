import React, { createContext, useContext, useEffect, useState } from 'react'

const AuthContext = createContext()


export function useAuth() {
    return useContext(AuthContext)
}

export function AuthProvider(props) {
    const [currentUser, setCurrentUser] = useState(()=>{
        return JSON.parse(localStorage.getItem("user")) || null
    })
    const [isLoggedIn, setIsLoggedIn] = useState(false)
    const value = {
        currentUser,
        setCurrentUser,
        isLoggedIn,
        setIsLoggedIn
    }

    // useEffect(()=>{
    //     setCurrentUser(JSON.parse(localStorage.getItem("user")))
    // },[])

    return (
        <AuthContext.Provider value={value}>
            {props.children}
        </AuthContext.Provider>
    )
}

