export default function FilmingGuide() {
  return (
    <section className="card">
      <h2>촬영 가이드</h2>
      <ul className="guide">
        <li>투수의 옆모습이 보이도록 투구 방향과 90° 각도로 촬영하세요.</li>
        <li>삼각대로 카메라를 고정하세요. 손으로 들고 찍으면 속도 값이 부정확해집니다.</li>
        <li>머리부터 발끝까지 전신이 프레임 안에 들어와야 합니다.</li>
        <li>녹화를 시작한 뒤 0.5초 이상 똑바로 선 자세로 있다가 투구하세요. 이 자세로 키 기준 크기를 맞춥니다.</li>
        <li>슬로모션(120~240fps)으로 촬영하면 더 정확합니다.</li>
        <li>
          슬로모션 영상은 원본 파일을 그대로 올리세요. 결과 화면의 &quot;분석 fps&quot;가 30처럼 실제 촬영 fps보다
          낮게 나오면, 올릴 때 &quot;촬영 fps&quot;에 촬영할 때의 fps(예: 240)를 입력하세요.
        </li>
      </ul>
    </section>
  );
}
