// const express = require("express");
// const cors = require("cors");

// const app = express()

// const mysql = require('mysql');

// var db = mysql.createConnection({
//     host     : 'localhost',
//     user     : 'root',
//     password : 'semih1306',
//     database : 'tasarim'
// });

// app.use(cors())
// app.use(express.json())


// app.get("/deneme",(req,res)=>{
//     console.log("deneme requesti")
//     const q = "select * from users"
//     db.query(q,(err,data)=>{
//         if(err) return res.json(err)
//         return res.json(data)
//     })
// })


// app.post("/",(req,res)=>{
//     const q = `insert into users (tcNo,password,healthId) values ?;`;

//     var user={
//         "tcNo":req.body.tcNo,
//         "password":req.body.password,
//         "healthId":req.body.healthId,
//       }

//     db.query(q,user,(err,data)=>{
//         if(err) return res.json(err)
//         return res.json(data)
//     })
// })



// const PORT = 5000
// app.listen(PORT,()=>{
//     console.log(`Server is running on ${PORT}`)
// })


// var fs = require('fs');
// var index = fs.readFileSync( 'index.html');

const express = require('express');
const { createServer } = require('http');
const { SerialPort } = require('serialport');
const { DelimiterParser } = require("@serialport/parser-delimiter")
const { Server } = require('socket.io');
const mysql = require('mysql');
const cors = require("cors")
const app = express();
const server = createServer(app);

app.use(cors())
app.use(express.json())
const io = new Server(server,{
    cors:{
        origin:"*",
        methods:["GET","POST"]
    }
});


var db = mysql.createConnection({
    host     : 'localhost',
    user     : 'root',
    password : 'semih1306',
    database : 'tasarim'
});


// var port = new SerialPort({
//     path: 'COM5',
//     baudRate: 9600,

// });

// const parser = port.pipe(new DelimiterParser({ delimiter: '\n' }));


io.on('connection',(socket)=>{
    console.log("Node is listening to port")
})

// parser.on('data', function (data) {
//     console.log('Received data from port: ' + data);
//     var enc = new TextDecoder();
//     var arr = new Uint8Array(data);
//     ready = enc.decode(arr)
//     io.emit('pulse', ready)
// });


app.get("/",(req,res)=>{
    const q = "SELECT * FROM users"
    db.query(q, (err,data)=>{
        if(err) return res.json(err)
        return res.json(data)
    })
})

app.post("/users",(req,res)=>{
    const { tcNo, password, healthId } = req.body;
    // Veritabanına yeni bir kayıt eklemek için sorgu
    const yeniKayit = {
      tcNo: tcNo,
      password: password,
      healthId: healthId
    };
    const q = 'INSERT INTO users SET ?'
    db.query(q, yeniKayit, (err,data)=>{
        if(err) return res.json(err)
        return res.json("Kayıt başarıyla eklendi")
    })
})


app.post("/register",(req,res)=>{
    const { ad,tcNo,soyad,sigara,yas,cinsiyet,sifre,diyabet,hipertansiyon,kanbasinci,felc,vki,sigaraGunde,mail } = req.body;
    // Veritabanına yeni bir kayıt eklemek için sorgu
    const yeniKayit = {
        ad,soyad,tcNo,sigara,yas,cinsiyet,sifre,diyabet,hipertansiyon,kanbasinci,felc,vki,sigaraGunde,mail
    };
    console.log(yeniKayit)
    const q = 'INSERT INTO customer SET ?'
    db.query(q, yeniKayit, (err,data)=>{
        if(err) return res.json(err)
        return res.json("Kayıt başarıyla eklendi")
    })
})

app.post("/login",(req,res)=>{
    const { tcNo,sifre } = req.body;
    // Veritabanına yeni bir kayıt eklemek için sorgu
    const q = 'SELECT * FROM customer WHERE tcNo = ? AND sifre = ?'
    console.log(tcNo,sifre)
    if(tcNo && sifre)
    {
        db.query(q, [tcNo,sifre], (err,data)=>{
            if(err) return res.json(err)
            if(data.length > 0){
                console.log(data)
                return res.json(data)
            }else{
                return res.status(400).json({message:'Kullanici adi veya şifre yanlış'})
            }
        })
    }else{
        return res.json("tc kimlik no veya şifre eksik")
    }
})



server.listen(4000, () => {
    console.log("Server running on port 4000")
});




