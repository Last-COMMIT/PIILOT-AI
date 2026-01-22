"""
이미지를 Base64로 변환하는 헬퍼 스크립트
Postman 테스트용
"""
import base64
import sys
import os


def image_to_base64(image_path: str) -> str:
    """
    이미지 파일을 Base64 문자열로 변환
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        Base64 인코딩된 문자열 (data:image/jpeg;base64,... 형식)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 파일 확장자로 MIME 타입 결정
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python image_to_base64.py <이미지_파일_경로>")
        print("\n예시:")
        print("  python image_to_base64.py test_image/input_image2.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        base64_string = image_to_base64(image_path)
        
        # 결과 출력
        print("\n✅ Base64 변환 완료!\n")
        print("=" * 80)
        print("Postman Request Body에 사용할 JSON:")
        print("=" * 80)
        print(f'{{')
        print(f'  "image_data": "{base64_string[:100]}...",')
        print(f'  "image_format": "base64"')
        print(f'}}')
        print("=" * 80)
        
        # 전체 Base64 문자열을 파일로 저장 (선택사항)
        output_file = f"{image_path}.base64.txt"
        with open(output_file, "w") as f:
            f.write(base64_string)
        print(f"\n📁 전체 Base64 문자열이 저장되었습니다: {output_file}")
        print(f"   (파일 크기: {len(base64_string)} bytes)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

