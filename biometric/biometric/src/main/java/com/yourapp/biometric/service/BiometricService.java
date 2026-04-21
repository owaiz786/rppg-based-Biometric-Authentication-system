package com.yourapp.biometric.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yourapp.biometric.model.User;
import com.yourapp.biometric.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
    private static final Logger log = LoggerFactory.getLogger(BiometricService.class);

    @Autowired
    private UserRepository userRepository;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    private static final String PYTHON_ANALYZE_URL = "http://localhost:8000/api/ml/analyze";
    private static final String PYTHON_FULL_URL    = "http://localhost:8000/api/ml/analyze-full";
    private static final String LOGIN_CHALLENGES   = "blink,head_turn";

    private static final double SIMILARITY_THRESHOLD = 0.75;

    // ── Enrollment ─────────────────────────────────────────────────────────────
    public String enrollUser(String username, MultipartFile video) throws Exception {
        log.info("Enrolling user: {}", username);

        JsonNode py = callPython(video, PYTHON_ANALYZE_URL, null);

        if (!py.path("success").asBoolean(false)) {
            throw new RuntimeException("Enrollment failed: " +
                py.path("spoof_reason").asText("Unknown error"));
        }

        JsonNode embNode = py.path("embedding");
        if (embNode.isMissingNode() || !embNode.isArray() || embNode.size() == 0) {
            throw new RuntimeException("No face embedding returned. Please face the camera directly.");
        }

        User user = userRepository.findByUsername(username).orElse(new User());
        user.setUsername(username);
        user.setFaceEmbedding(objectMapper.writeValueAsString(embNode));
        userRepository.save(user);

        log.info("Enrolled user {} with embedding size {}", username, embNode.size());
        return "User enrolled successfully!";
    }

    // ── Login ──────────────────────────────────────────────────────────────────
    public boolean authenticateUser(String username, MultipartFile video) throws Exception {
        log.info("Authenticating user: {}", username);

        Optional<User> userOpt = userRepository.findByUsername(username);
        if (userOpt.isEmpty()) {
            throw new RuntimeException("User '" + username + "' not found. Please enroll first.");
        }

        JsonNode py = callPython(video, PYTHON_FULL_URL, LOGIN_CHALLENGES);

        // FIX: Respect the liveness decision from Python.
        // The previous code only blocked on coherence > 0.98 and let everything
        // else through to face matching. This meant a spoof that passed even
        // one weak liveness layer would proceed to face matching.
        // Now: if the ML service returns success=false for ANY reason other
        // than an internal server error, we block — the Python pipeline
        // already has its own forgiving logic internally.
        boolean isReal = py.path("success").asBoolean(false);

        if (!isReal) {
            String reason    = py.path("spoof_reason").asText("Liveness check failed");
            double coherence = py.path("coherence_score").asDouble(0.0);
            log.warn("Liveness FAILED for {} — coherence={}, reason: {}", username, coherence, reason);
            // Return false to trigger 401 in the controller
            return false;
        }

        // Log all three liveness layer results for debugging
        double  coherence   = py.path("coherence_score").asDouble(0.0);
        boolean chalPassed  = py.path("challenge_passed").asBoolean(false);
        boolean bcgPassed   = py.path("bcg_passed").asBoolean(false);
        double  bcgHr       = py.path("bcg_hr_bpm").asDouble(0.0);
        double  rppgHr      = py.path("rppg_hr_bpm").asDouble(0.0);

        log.info("Liveness PASSED — coherence:{}, challenge:{}, BCG:{} (BCG={}bpm, rPPG={}bpm)",
                coherence, chalPassed, bcgPassed, bcgHr, rppgHr);

        // ── Face matching ──────────────────────────────────────────────────────
        JsonNode embNode = py.path("embedding");
        if (embNode.isMissingNode() || !embNode.isArray() || embNode.size() == 0) {
            throw new RuntimeException("No face embedding returned from ML service.");
        }

        List<Double> stored  = objectMapper.readValue(
            userOpt.get().getFaceEmbedding(), new TypeReference<>() {}
        );
        List<Double> current = objectMapper.readValue(
            objectMapper.writeValueAsString(embNode), new TypeReference<>() {}
        );

        double similarity = cosineSim(stored, current);
        log.info("Face similarity for {}: {} (threshold={})", username, similarity, SIMILARITY_THRESHOLD);

        return similarity >= SIMILARITY_THRESHOLD;
    }

    // ── HTTP helper ────────────────────────────────────────────────────────────
    private JsonNode callPython(MultipartFile video, String url, String challenges) throws Exception {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", video.getResource());
        if (challenges != null && !challenges.isBlank()) {
            body.add("challenges", challenges);
        }

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
            String responseBody = response.getBody();
            if (responseBody == null)
                throw new RuntimeException("Empty response from ML service at " + url);
            return objectMapper.readTree(responseBody);
        } catch (HttpClientErrorException e) {
            try {
                JsonNode errorJson = objectMapper.readTree(e.getResponseBodyAsString());
                // For 401, Python already decided it's a spoof — pass that back
                if (e.getStatusCode() == HttpStatus.UNAUTHORIZED) {
                    log.warn("ML service returned 401 (spoof detected): {}",
                        errorJson.path("spoof_reason").asText("Unknown"));
                    return errorJson;  // success=false is already set in the body
                }
                throw new RuntimeException("ML service error (" + e.getStatusCode() + "): " +
                    errorJson.path("spoof_reason").asText(e.getResponseBodyAsString()));
            } catch (Exception ignored) {
                throw new RuntimeException("ML service error (" + e.getStatusCode() + "): " +
                    e.getResponseBodyAsString());
            }
        }
    }

    // ── Cosine similarity ──────────────────────────────────────────────────────
    private double cosineSim(List<Double> a, List<Double> b) {
        if (a == null || b == null || a.size() != b.size() || a.isEmpty())
            return 0.0;

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            na  += a.get(i) * a.get(i);
            nb  += b.get(i) * b.get(i);
        }
        return (na == 0 || nb == 0) ? 0.0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}