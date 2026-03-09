package com.yourapp.biometric.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yourapp.biometric.dto.PythonResponse;
import com.yourapp.biometric.model.User;
import com.yourapp.biometric.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Optional;

@Service
public class BiometricService {

    @Autowired
    private UserRepository userRepository;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    // FIX: correct Python ML service endpoint
    // Python's /api/ml/analyze accepts field name "file" and returns
    // { success, is_real, spoof_reason, coherence_score, embedding }
    private final String PYTHON_URL = "http://localhost:8000/api/ml/analyze";

    public String enrollUser(String username, MultipartFile video) throws Exception {
        // Check if user already enrolled
        Optional<User> existing = userRepository.findByUsername(username);

        PythonResponse pyResponse = sendToPython(video);

        // FIX: during enrollment, only block on actual spoof (screen replay).
        // "Weak signal" is not a spoof — it just means poor camera conditions.
        // We still have a valid embedding in that case.
        if (!pyResponse.isSuccess()) {
            throw new RuntimeException("ML service error: " + pyResponse.getSpoofReason());
        }

        if (pyResponse.getEmbedding() == null || pyResponse.getEmbedding().isEmpty()) {
            throw new RuntimeException("No face embedding returned. Please face the camera directly.");
        }

        // Upsert: update embedding if user exists, otherwise create new record
        User user = existing.orElse(new User());
        user.setUsername(username);
        user.setFaceEmbedding(objectMapper.writeValueAsString(pyResponse.getEmbedding()));
        userRepository.save(user);   // saves to Neon PostgreSQL via JPA

        return "User enrolled successfully! Embedding stored in database.";
    }

    public boolean authenticateUser(String username, MultipartFile video) throws Exception {
        // Retrieve stored embedding from Neon PostgreSQL
        Optional<User> userOpt = userRepository.findByUsername(username);
        if (userOpt.isEmpty()) {
            throw new RuntimeException("User not found. Please enroll first.");
        }

        PythonResponse pyResponse = sendToPython(video);

        if (!pyResponse.isSuccess()) {
            throw new RuntimeException("ML service error: " + pyResponse.getSpoofReason());
        }

        // FIX: block on screen replay (coherence > 0.95), not weak signal
        // Python already filters screen replay before returning success=true,
        // but double-check here as well
        if (!pyResponse.isReal() && pyResponse.getCoherenceScore() > 0.95) {
            return false;  // definite spoof
        }

        if (pyResponse.getEmbedding() == null || pyResponse.getEmbedding().isEmpty()) {
            throw new RuntimeException("No face embedding returned from ML service.");
        }

        // Compare current embedding against stored embedding from Neon
        List<Double> storedEmbedding = objectMapper.readValue(
                userOpt.get().getFaceEmbedding(), new TypeReference<>() {});

        double similarity = calculateCosineSimilarity(storedEmbedding, pyResponse.getEmbedding());
        return similarity > 0.75;
    }

    /**
     * Sends video to Python ML service.
     * Field name must be "file" — matches Python FastAPI endpoint parameter.
     */
    private PythonResponse sendToPython(MultipartFile video) throws Exception {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", video.getResource());  // "file" matches Python's File(...) param name

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<PythonResponse> response = restTemplate.postForEntity(
                    PYTHON_URL, requestEntity, PythonResponse.class);

            if (response.getBody() == null) {
                throw new RuntimeException("Empty response from ML service");
            }
            return response.getBody();

        } catch (HttpClientErrorException e) {
            // Parse error body from Python so we get the real reason
            String errorBody = e.getResponseBodyAsString();
            try {
                PythonResponse errResp = objectMapper.readValue(errorBody, PythonResponse.class);
                return errResp;
            } catch (Exception ignored) {
                throw new RuntimeException("ML service error (" + e.getStatusCode() + "): " + errorBody);
            }
        }
    }

    private double calculateCosineSimilarity(List<Double> vectorA, List<Double> vectorB) {
        if (vectorA == null || vectorB == null || vectorA.size() != vectorB.size()) {
            return 0.0;
        }
        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vectorA.size(); i++) {
            dotProduct += vectorA.get(i) * vectorB.get(i);
            normA += Math.pow(vectorA.get(i), 2);
            normB += Math.pow(vectorB.get(i), 2);
        }
        if (normA == 0 || normB == 0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}