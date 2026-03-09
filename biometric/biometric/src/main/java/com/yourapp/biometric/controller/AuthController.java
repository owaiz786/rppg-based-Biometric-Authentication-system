package com.yourapp.biometric.controller;

import com.yourapp.biometric.service.BiometricService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "http://localhost:3000") // Connects to your Next.js frontend
public class AuthController {

    @Autowired
    private BiometricService biometricService;

    @PostMapping("/enroll")
    public ResponseEntity<?> enroll(
            @RequestParam("username") String username,
            @RequestParam("video") MultipartFile video) {
        try {
            String message = biometricService.enrollUser(username, video);
            return ResponseEntity.ok(Map.of("success", true, "message", message));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("success", false, "message", e.getMessage()));
        }
    }

    @PostMapping("/login-video")
    public ResponseEntity<?> login(
            @RequestParam("username") String username,
            @RequestParam("video") MultipartFile video) {
        try {
            boolean isMatch = biometricService.authenticateUser(username, video);
            Map<String, Object> response = new HashMap<>();
            
            if (isMatch) {
                response.put("success", true);
                response.put("message", "Authentication Successful.");
                response.put("token", "mock_jwt_token_12345"); 
                return ResponseEntity.ok(response);
            } else {
                response.put("success", false);
                response.put("message", "Spoof detected or face mismatch.");
                return ResponseEntity.status(401).body(response);
            }
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("success", false, "message", e.getMessage()));
        }
    }
}